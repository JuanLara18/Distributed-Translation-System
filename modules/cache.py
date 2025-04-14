#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache module for the distributed translation system.
Handles caching of translations to avoid redundant API calls.
"""

import os
import time
import json
import sqlite3
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Set
import logging
import threading
import hashlib


class AbstractCache(ABC):
    """
    Abstract base class for cache implementations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cache.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ttl = config.get('cache', {}).get('ttl', 2592000)  # Default 30 days in seconds
    
    @abstractmethod
    def get(self, source_text: str, source_language: str, target_language: str) -> Optional[str]:
        """
        Get a translation from the cache.
        
        Args:
            source_text: Original text to look up
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Cached translation or None if not found
        """
        pass
    
    @abstractmethod
    def set(self, source_text: str, source_language: str, target_language: str, 
            translation: str) -> bool:
        """
        Store a translation in the cache.
        
        Args:
            source_text: Original text
            source_language: Source language code
            target_language: Target language code
            translation: Translated text to store
            
        Returns:
            Boolean indicating success
        """
        pass
    
    @abstractmethod
    def batch_get(self, items: List[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], Optional[str]]:
        """
        Get multiple translations from the cache in one operation.
        
        Args:
            items: List of (source_text, source_language, target_language) tuples
            
        Returns:
            Dictionary mapping each input tuple to its cached translation (or None)
        """
        pass
    
    @abstractmethod
    def batch_set(self, translations: Dict[Tuple[str, str, str], str]) -> int:
        """
        Store multiple translations in the cache in one operation.
        
        Args:
            translations: Dictionary mapping (source_text, source_language, target_language)
                          tuples to their translations
                          
        Returns:
            Number of items successfully cached
        """
        pass
    
    @abstractmethod
    def contains(self, source_text: str, source_language: str, target_language: str) -> bool:
        """
        Check if a translation exists in the cache.
        
        Args:
            source_text: Original text to check
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Boolean indicating if the translation is cached
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> int:
        """
        Remove expired entries from the cache.
        
        Returns:
            Number of items removed
        """
        pass
    
    @abstractmethod
    def flush(self) -> bool:
        """
        Ensure all pending writes are committed to persistent storage.
        
        Returns:
            Boolean indicating success
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all entries from the cache.
        
        Returns:
            Boolean indicating success
        """
        pass
    
    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        pass


class SQLiteCache(AbstractCache):
    """
    SQLite-based implementation of the cache.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SQLite cache.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.location = config.get('cache', {}).get('location', './cache/translations.db')
        self.conn = None
        self.cursor = None
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """
        Initialize the SQLite database and create needed tables if they don't exist.
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.location)), exist_ok=True)
            
            # Connect to the database with WAL journal mode for better concurrency
            self.conn = sqlite3.connect(self.location, check_same_thread=False, timeout=30.0)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.cursor = self.conn.cursor()
            
            # Create the translations table if it doesn't exist
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                id TEXT PRIMARY KEY,
                source_text TEXT,
                source_language TEXT,
                target_language TEXT,
                translation TEXT,
                timestamp INTEGER
            )
            ''')
            
            # Create indexes for faster lookups
            self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_translation_lookup ON translations 
            (source_language, target_language, source_text)
            ''')
            
            self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON translations (timestamp)
            ''')
            
            self.conn.commit()
            self.logger.info(f"SQLite cache initialized at {self.location}")
        except sqlite3.Error as e:
            self.logger.error(f"Error initializing SQLite database: {str(e)}")
            raise
    
    def _generate_id(self, source_text: str, source_language: str, target_language: str) -> str:
        """
        Generate a unique ID for a translation entry.
        
        Args:
            source_text: Original text
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Unique ID string
        """
        key = f"{source_language}:{target_language}:{source_text}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def get(self, source_text: str, source_language: str, target_language: str) -> Optional[str]:
        """
        Get a translation from the cache.
        
        Args:
            source_text: Original text to look up
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Cached translation or None if not found
        """
        try:
            current_time = int(time.time())
            expiry_time = current_time - self.ttl
            entry_id = self._generate_id(source_text, source_language, target_language)
            
            self.cursor.execute(
                '''
                SELECT translation FROM translations 
                WHERE id = ? AND timestamp > ?
                ''',
                (entry_id, expiry_time)
            )
            
            result = self.cursor.fetchone()
            if result:
                return result[0]
            return None
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving from cache: {str(e)}")
            return None
    
    def set(self, source_text: str, source_language: str, target_language: str, 
            translation: str) -> bool:
        """
        Store a translation in the cache.
        
        Args:
            source_text: Original text
            source_language: Source language code
            target_language: Target language code
            translation: Translated text to store
            
        Returns:
            Boolean indicating success
        """
        try:
            current_time = int(time.time())
            entry_id = self._generate_id(source_text, source_language, target_language)
            
            self.cursor.execute(
                '''
                INSERT OR REPLACE INTO translations 
                (id, source_text, source_language, target_language, translation, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (entry_id, source_text, source_language, target_language, translation, current_time)
            )
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            self.logger.error(f"Error setting cache: {str(e)}")
            return False
    
    def batch_get(self, items: List[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], Optional[str]]:
        """
        Get multiple translations from the cache in one operation.
        
        Args:
            items: List of (source_text, source_language, target_language) tuples
            
        Returns:
            Dictionary mapping each input tuple to its cached translation (or None)
        """
        results = {}
        if not items:
            return results
            
        try:
            current_time = int(time.time())
            expiry_time = current_time - self.ttl
            
            # Generate IDs for all items
            ids = [self._generate_id(text, src_lang, tgt_lang) for text, src_lang, tgt_lang in items]
            
            # Build a parameterized query with placeholders
            placeholders = ','.join(['?'] * len(ids))
            query = f'''
            SELECT id, translation 
            FROM translations 
            WHERE id IN ({placeholders})
            AND timestamp > ?
            '''
            
            # Execute query with all IDs and expiry time
            params = ids + [expiry_time]
            self.cursor.execute(query, params)
            
            # Create a mapping from ID to translation
            id_to_translation = {row[0]: row[1] for row in self.cursor.fetchall()}
            
            # Map each input item to its translation (or None)
            for i, (text, src_lang, tgt_lang) in enumerate(items):
                entry_id = ids[i]
                results[(text, src_lang, tgt_lang)] = id_to_translation.get(entry_id)
                    
            return results
        except sqlite3.Error as e:
            self.logger.error(f"Error in batch get: {str(e)}")
            # Return None for all items on error
            return {item: None for item in items}
    
    def batch_set(self, translations: Dict[Tuple[str, str, str], str]) -> int:
        """
        Store multiple translations in the cache in one operation.
        
        Args:
            translations: Dictionary mapping (source_text, source_language, target_language)
                          tuples to their translations
                          
        Returns:
            Number of items successfully cached
        """
        if not translations:
            return 0
            
        try:
            current_time = int(time.time())
            
            # Begin a transaction for efficiency
            self.conn.execute("BEGIN TRANSACTION")
            
            count = 0
            for (source_text, source_language, target_language), translation in translations.items():
                entry_id = self._generate_id(source_text, source_language, target_language)
                self.cursor.execute(
                    '''
                    INSERT OR REPLACE INTO translations 
                    (id, source_text, source_language, target_language, translation, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''',
                    (entry_id, source_text, source_language, target_language, translation, current_time)
                )
                count += 1
            
            # Commit the transaction
            self.conn.commit()
            return count
        except sqlite3.Error as e:
            self.logger.error(f"Error in batch set: {str(e)}")
            # Rollback on error
            try:
                self.conn.rollback()
            except:
                pass
            return 0
    
    def contains(self, source_text: str, source_language: str, target_language: str) -> bool:
        """
        Check if a translation exists in the cache.
        
        Args:
            source_text: Original text to check
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Boolean indicating if the translation is cached
        """
        try:
            current_time = int(time.time())
            expiry_time = current_time - self.ttl
            entry_id = self._generate_id(source_text, source_language, target_language)
            
            self.cursor.execute(
                '''
                SELECT 1 FROM translations 
                WHERE id = ? AND timestamp > ?
                LIMIT 1
                ''',
                (entry_id, expiry_time)
            )
            
            return self.cursor.fetchone() is not None
        except sqlite3.Error as e:
            self.logger.error(f"Error checking cache: {str(e)}")
            return False
    
    def cleanup(self) -> int:
        """
        Remove expired entries from the cache.
        
        Returns:
            Number of items removed
        """
        try:
            current_time = int(time.time())
            expiry_time = current_time - self.ttl
            
            self.cursor.execute(
                "DELETE FROM translations WHERE timestamp <= ?",
                (expiry_time,)
            )
            
            count = self.cursor.rowcount
            self.conn.commit()
            self.logger.info(f"Removed {count} expired cache entries")
            return count
        except sqlite3.Error as e:
            self.logger.error(f"Error cleaning up cache: {str(e)}")
            return 0
    
    def flush(self) -> bool:
        """
        Ensure all pending writes are committed to persistent storage.
        
        Returns:
            Boolean indicating success
        """
        try:
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            self.logger.error(f"Error flushing cache: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all entries from the cache.
        
        Returns:
            Boolean indicating success
        """
        try:
            self.cursor.execute("DELETE FROM translations")
            self.conn.commit()
            self.logger.info("Cache cleared")
            return True
        except sqlite3.Error as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'total_entries': 0,
            'valid_entries': 0,
            'expired_entries': 0,
            'size_bytes': 0,
            'languages': {
                'source': [],
                'target': []
            }
        }
        
        try:
            # Get total entries
            self.cursor.execute("SELECT COUNT(*) FROM translations")
            stats['total_entries'] = self.cursor.fetchone()[0]
            
            # Get valid entries
            current_time = int(time.time())
            expiry_time = current_time - self.ttl
            
            self.cursor.execute(
                "SELECT COUNT(*) FROM translations WHERE timestamp > ?",
                (expiry_time,)
            )
            stats['valid_entries'] = self.cursor.fetchone()[0]
            
            # Calculate expired entries
            stats['expired_entries'] = stats['total_entries'] - stats['valid_entries']
            
            # Get languages
            self.cursor.execute(
                "SELECT DISTINCT source_language, target_language FROM translations"
            )
            language_pairs = self.cursor.fetchall()
            source_languages = {lang[0] for lang in language_pairs}
            target_languages = {lang[1] for lang in language_pairs}
            stats['languages'] = {
                'source': list(source_languages),
                'target': list(target_languages)
            }
            
            # Estimate size
            self.cursor.execute(
                "SELECT SUM(LENGTH(source_text)) + SUM(LENGTH(translation)) FROM translations"
            )
            size_result = self.cursor.fetchone()
            if size_result and size_result[0]:
                stats['size_bytes'] = size_result[0]
            
            return stats
        except sqlite3.Error as e:
            self.logger.error(f"Error getting cache stats: {str(e)}")
            return stats
    
    def __del__(self):
        """
        Destructor to ensure proper cleanup of database connections.
        """
        try:
            if self.conn:
                self.conn.close()
        except:
            pass


class PostgresCache(AbstractCache):
    """
    PostgreSQL-based implementation of the cache.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PostgreSQL cache.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.connection_string = config.get('cache', {}).get('connection_string')
        if not self.connection_string:
            raise ValueError("PostgreSQL connection string not provided in config")
        
        # Import psycopg2 here to avoid dependency if not used
        try:
            import psycopg2
            import psycopg2.extras
            self.psycopg2 = psycopg2
        except ImportError:
            self.logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
            raise
            
        self.conn = None
        self.cursor = None
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """
        Initialize the PostgreSQL database and create needed tables if they don't exist.
        """
        try:
            self.conn = self.psycopg2.connect(self.connection_string)
            self.cursor = self.conn.cursor()
            
            # Create the translations table if it doesn't exist
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                id TEXT PRIMARY KEY,
                source_text TEXT,
                source_language TEXT,
                target_language TEXT,
                translation TEXT,
                timestamp INTEGER
            )
            ''')
            
            # Create indexes for faster lookups
            self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_translation_lookup ON translations 
            (source_language, target_language, source_text)
            ''')
            
            self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON translations (timestamp)
            ''')
            
            self.conn.commit()
            self.logger.info("PostgreSQL cache initialized")
        except Exception as e:
            self.logger.error(f"Error initializing PostgreSQL database: {str(e)}")
            raise
    
    def _generate_id(self, source_text: str, source_language: str, target_language: str) -> str:
        """
        Generate a unique ID for a translation entry.
        
        Args:
            source_text: Original text
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Unique ID string
        """
        key = f"{source_language}:{target_language}:{source_text}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def get(self, source_text: str, source_language: str, target_language: str) -> Optional[str]:
        """
        Get a translation from the cache.
        
        Args:
            source_text: Original text to look up
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Cached translation or None if not found
        """
        try:
            current_time = int(time.time())
            expiry_time = current_time - self.ttl
            entry_id = self._generate_id(source_text, source_language, target_language)
            
            self.cursor.execute(
                '''
                SELECT translation FROM translations 
                WHERE id = %s AND timestamp > %s
                ''',
                (entry_id, expiry_time)
            )
            
            result = self.cursor.fetchone()
            if result:
                return result[0]
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving from PostgreSQL cache: {str(e)}")
            return None
    
    def set(self, source_text: str, source_language: str, target_language: str, 
            translation: str) -> bool:
        """
        Store a translation in the cache.
        
        Args:
            source_text: Original text
            source_language: Source language code
            target_language: Target language code
            translation: Translated text to store
            
        Returns:
            Boolean indicating success
        """
        try:
            current_time = int(time.time())
            entry_id = self._generate_id(source_text, source_language, target_language)
            
            self.cursor.execute(
                '''
                INSERT INTO translations 
                (id, source_text, source_language, target_language, translation, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) 
                DO UPDATE SET translation = EXCLUDED.translation, timestamp = EXCLUDED.timestamp
                ''',
                (entry_id, source_text, source_language, target_language, translation, current_time)
            )
            
            self.conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error setting PostgreSQL cache: {str(e)}")
            return False
    
    def batch_get(self, items: List[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], Optional[str]]:
        """
        Get multiple translations from the cache in one operation.
        
        Args:
            items: List of (source_text, source_language, target_language) tuples
            
        Returns:
            Dictionary mapping each input tuple to its cached translation (or None)
        """
        results = {}
        if not items:
            return results
            
        try:
            current_time = int(time.time())
            expiry_time = current_time - self.ttl
            
            # Generate IDs for all items
            ids = [self._generate_id(text, src_lang, tgt_lang) for text, src_lang, tgt_lang in items]
            id_to_item = {self._generate_id(text, src_lang, tgt_lang): (text, src_lang, tgt_lang) 
                         for text, src_lang, tgt_lang in items}
            
            # Using ANY to match multiple IDs in PostgreSQL
            self.cursor.execute(
                '''
                SELECT id, translation 
                FROM translations 
                WHERE id = ANY(%s)
                AND timestamp > %s
                ''',
                (ids, expiry_time)
            )
            
            # Map query results back to original items
            for row in self.cursor.fetchall():
                entry_id, translation = row
                if entry_id in id_to_item:
                    results[id_to_item[entry_id]] = translation
            
            # Add None for missing items
            for item in items:
                if item not in results:
                    results[item] = None
                    
            return results
        except Exception as e:
            self.logger.error(f"Error in PostgreSQL batch get: {str(e)}")
            # Return None for all items on error
            return {item: None for item in items}
    
    def batch_set(self, translations: Dict[Tuple[str, str, str], str]) -> int:
        """
        Store multiple translations in the cache in one operation.
        
        Args:
            translations: Dictionary mapping (source_text, source_language, target_language)
                          tuples to their translations
                          
        Returns:
            Number of items successfully cached
        """
        if not translations:
            return 0
            
        try:
            current_time = int(time.time())
            
            # Begin a transaction
            with self.conn:
                values = []
                for (source_text, source_language, target_language), translation in translations.items():
                    entry_id = self._generate_id(source_text, source_language, target_language)
                    values.append((entry_id, source_text, source_language, target_language, translation, current_time))
                
                # Use execute_values for efficient batch insert
                self.psycopg2.extras.execute_values(
                    self.cursor,
                    '''
                    INSERT INTO translations 
                    (id, source_text, source_language, target_language, translation, timestamp)
                    VALUES %s
                    ON CONFLICT (id) 
                    DO UPDATE SET translation = EXCLUDED.translation, timestamp = EXCLUDED.timestamp
                    ''',
                    values
                )
                
                return len(values)
        except Exception as e:
            self.logger.error(f"Error in PostgreSQL batch set: {str(e)}")
            return 0
    
    def contains(self, source_text: str, source_language: str, target_language: str) -> bool:
        """
        Check if a translation exists in the cache.
        
        Args:
            source_text: Original text to check
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Boolean indicating if the translation is cached
        """
        try:
            current_time = int(time.time())
            expiry_time = current_time - self.ttl
            entry_id = self._generate_id(source_text, source_language, target_language)
            
            self.cursor.execute(
                '''
                SELECT 1 FROM translations 
                WHERE id = %s AND timestamp > %s
                LIMIT 1
                ''',
                (entry_id, expiry_time)
            )
            
            return self.cursor.fetchone() is not None
        except Exception as e:
            self.logger.error(f"Error checking PostgreSQL cache: {str(e)}")
            return False
    
    def cleanup(self) -> int:
        """
        Remove expired entries from the cache.
        
        Returns:
            Number of items removed
        """
        try:
            current_time = int(time.time())
            expiry_time = current_time - self.ttl
            
            self.cursor.execute(
                "DELETE FROM translations WHERE timestamp <= %s",
                (expiry_time,)
            )
            
            count = self.cursor.rowcount
            self.conn.commit()
            self.logger.info(f"Removed {count} expired cache entries")
            return count
        except Exception as e:
            self.logger.error(f"Error cleaning up PostgreSQL cache: {str(e)}")
            return 0
    
    def flush(self) -> bool:
        """
        Ensure all pending writes are committed to persistent storage.
        
        Returns:
            Boolean indicating success
        """
        try:
            self.conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error flushing PostgreSQL cache: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all entries from the cache.
        
        Returns:
            Boolean indicating success
        """
        try:
            self.cursor.execute("TRUNCATE TABLE translations")
            self.conn.commit()
            self.logger.info("Cache cleared")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing PostgreSQL cache: {str(e)}")
            return False
    
    def stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'total_entries': 0,
            'valid_entries': 0,
            'expired_entries': 0,
            'size_bytes': 0,
            'languages': {
                'source': [],
                'target': []
            }
        }
        
        try:
            # Get total entries
            self.cursor.execute("SELECT COUNT(*) FROM translations")
            stats['total_entries'] = self.cursor.fetchone()[0]
            
            # Get valid entries
            current_time = int(time.time())
            expiry_time = current_time - self.ttl
            
            self.cursor.execute(
                "SELECT COUNT(*) FROM translations WHERE timestamp > %s",
                (expiry_time,)
            )
            stats['valid_entries'] = self.cursor.fetchone()[0]
            
            # Calculate expired entries
            stats['expired_entries'] = stats['total_entries'] - stats['valid_entries']
            
            # Get languages
            self.cursor.execute(
                "SELECT DISTINCT source_language, target_language FROM translations"
            )
            language_pairs = self.cursor.fetchall()
            source_languages = {lang[0] for lang in language_pairs}
            target_languages = {lang[1] for lang in language_pairs}
            stats['languages'] = {
                'source': list(source_languages),
                'target': list(target_languages)
            }
            
            # Estimate size
            self.cursor.execute(
                "SELECT pg_total_relation_size('translations')"
            )
            size_result = self.cursor.fetchone()
            if size_result and size_result[0]:
                stats['size_bytes'] = size_result[0]
            
            return stats
        except Exception as e:
            self.logger.error(f"Error getting PostgreSQL cache stats: {str(e)}")
            return stats
    
    def __del__(self):
        """
        Destructor to ensure proper cleanup of database connections.
        """
        try:
            if self.conn:
                self.conn.close()
        except:
            pass


class MemoryCache(AbstractCache):
    """
    In-memory implementation of the cache for testing or small workloads.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the memory cache.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.cache = {}
        self.logger.info("Memory cache initialized")
    
    def get(self, source_text: str, source_language: str, target_language: str) -> Optional[str]:
        """
        Get a translation from the cache.
        
        Args:
            source_text: Original text to look up
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Cached translation or None if not found
        """
        key = (source_text, source_language, target_language)
        entry = self.cache.get(key)
        
        if not entry:
            return None
            
        timestamp, translation = entry
        current_time = int(time.time())
        
        if current_time - timestamp > self.ttl:
            # Entry expired
            return None
            
        return translation
    
    def set(self, source_text: str, source_language: str, target_language: str, 
            translation: str) -> bool:
        """
        Store a translation in the cache.
        
        Args:
            source_text: Original text
            source_language: Source language code
            target_language: Target language code
            translation: Translated text to store
            
        Returns:
            Boolean indicating success
        """
        key = (source_text, source_language, target_language)
        self.cache[key] = (int(time.time()), translation)
        return True
    
    def batch_get(self, items: List[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], Optional[str]]:
        """
        Get multiple translations from the cache in one operation.
        
        Args:
            items: List of (source_text, source_language, target_language) tuples
            
        Returns:
            Dictionary mapping each input tuple to its cached translation (or None)
        """
        results = {}
        current_time = int(time.time())
        
        for item in items:
            entry = self.cache.get(item)
            if entry and (current_time - entry[0] <= self.ttl):
                results[item] = entry[1]
            else:
                results[item] = None
                
        return results
    
    def batch_set(self, translations: Dict[Tuple[str, str, str], str]) -> int:
        """
        Store multiple translations in the cache in one operation.
        
        Args:
            translations: Dictionary mapping (source_text, source_language, target_language)
                          tuples to their translations
                          
        Returns:
            Number of items successfully cached
        """
        current_time = int(time.time())
        
        for key, translation in translations.items():
            self.cache[key] = (current_time, translation)
            
        return len(translations)
    
    def contains(self, source_text: str, source_language: str, target_language: str) -> bool:
        """
        Check if a translation exists in the cache.
        
        Args:
            source_text: Original text to check
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Boolean indicating if the translation is cached
        """
        key = (source_text, source_language, target_language)
        entry = self.cache.get(key)
        
        if not entry:
            return False
            
        timestamp = entry[0]
        current_time = int(time.time())
        
        return current_time - timestamp <= self.ttl
    
    def cleanup(self) -> int:
        """
        Remove expired entries from the cache.
        
        Returns:
            Number of items removed
        """
        current_time = int(time.time())
        expired_keys = []
        
        for key, (timestamp, _) in self.cache.items():
            if current_time - timestamp > self.ttl:
                expired_keys.append(key)
                
        for key in expired_keys:
            del self.cache[key]
            
        return len(expired_keys)
    
    def flush(self) -> bool:
        """
        Ensure all pending writes are committed to persistent storage.
        Memory cache doesn't need flushing, so always returns True.
        
        Returns:
            Boolean indicating success
        """
        return True
    
    def clear(self) -> bool:
        """
        Clear all entries from the cache.
        
        Returns:
            Boolean indicating success
        """
        self.cache.clear()
        return True
    
    def stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        current_time = int(time.time())
        total = len(self.cache)
        valid = sum(1 for timestamp, _ in self.cache.values() if current_time - timestamp <= self.ttl)
        
        source_languages = set()
        target_languages = set()
        size_bytes = 0
        
        for (text, src_lang, tgt_lang), (_, translation) in self.cache.items():
            source_languages.add(src_lang)
            target_languages.add(tgt_lang)
            size_bytes += len(text.encode('utf-8')) + len(translation.encode('utf-8'))
        
        return {
            'total_entries': total,
            'valid_entries': valid,
            'expired_entries': total - valid,
            'size_bytes': size_bytes,
            'languages': {
                'source': list(source_languages),
                'target': list(target_languages)
            }
        }


class CacheManager:
    """
    Manages caching operations and coordinates between different cache implementations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cache manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cache_type = config.get('cache', {}).get('type', 'sqlite')
        self.cache = self._create_cache(self.cache_type)
        
        # Stats tracking
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()  # Reentrant lock for thread safety
    
    def _create_cache(self, cache_type: str) -> AbstractCache:
        """
        Create and initialize the appropriate cache implementation.
        
        Args:
            cache_type: Type of cache to create
            
        Returns:
            Initialized cache implementation
            
        Raises:
            ValueError: If specified cache type is not supported
        """
        self.logger.info(f"Initializing {cache_type} cache")
        
        if cache_type.lower() == 'sqlite':
            return SQLiteCache(self.config)
        elif cache_type.lower() == 'postgres':
            return PostgresCache(self.config)
        elif cache_type.lower() == 'memory':
            return MemoryCache(self.config)
        else:
            error_msg = f"Unsupported cache type: {cache_type}. Use 'sqlite', 'postgres', or 'memory'."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def get(self, source_text: str, source_language: str, target_language: str) -> Optional[str]:
        """
        Get a translation from the cache.
        
        Args:
            source_text: Original text to look up
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Cached translation or None if not found
        """
        with self.lock:
            result = self.cache.get(source_text, source_language, target_language)
            if result:
                self.hit_count += 1
                self.logger.debug(f"Cache hit for: {source_language} -> {target_language}, text length: {len(source_text)}")
            else:
                self.miss_count += 1
                self.logger.debug(f"Cache miss for: {source_language} -> {target_language}, text length: {len(source_text)}")
            return result
    
    def set(self, source_text: str, source_language: str, target_language: str, 
            translation: str) -> bool:
        """
        Store a translation in the cache.
        
        Args:
            source_text: Original text
            source_language: Source language code
            target_language: Target language code
            translation: Translated text to store
            
        Returns:
            Boolean indicating success
        """
        with self.lock:
            success = self.cache.set(source_text, source_language, target_language, translation)
            if success:
                self.logger.debug(f"Cached: {source_language} -> {target_language}, text length: {len(source_text)}")
            else:
                self.logger.warning(f"Failed to cache: {source_language} -> {target_language}, text length: {len(source_text)}")
            return success
    
    def batch_get(self, texts: List[str], source_language: str, 
                target_language: str) -> Dict[str, Optional[str]]:
        """
        Get multiple translations from the cache.
        
        Args:
            texts: List of source texts
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Dictionary mapping source texts to translations (or None if not found)
        """
        if not texts:
            return {}
            
        with self.lock:
            # Create a list of tuples for batch_get
            items = [(text, source_language, target_language) for text in texts]
            
            # Get the results from the cache
            results_by_tuple = self.cache.batch_get(items)
            
            # Convert to a mapping from source text to translation
            results = {}
            for text in texts:
                key = (text, source_language, target_language)
                if key in results_by_tuple:
                    results[text] = results_by_tuple[key]
                    if results[text] is not None:
                        self.hit_count += 1
                    else:
                        self.miss_count += 1
                else:
                    results[text] = None
                    self.miss_count += 1
            
            hit_count = sum(1 for value in results.values() if value is not None)
            miss_count = len(texts) - hit_count
            
            self.logger.debug(f"Batch cache lookup: {hit_count} hits, {miss_count} misses")
            
            return results
    
    def batch_set(self, texts: List[str], translations: List[str], 
                source_language: str, target_language: str) -> int:
        """
        Store multiple translations in the cache.
        
        Args:
            texts: List of source texts
            translations: List of translated texts
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Number of items successfully cached
        """
        if not texts or not translations or len(texts) != len(translations):
            self.logger.warning("Invalid batch_set parameters: texts and translations lists must be non-empty and of the same length")
            return 0
            
        with self.lock:
            # Create a dictionary of tuples to translations
            translation_dict = {
                (text, source_language, target_language): trans
                for text, trans in zip(texts, translations)
            }
            
            # Set the translations in the cache
            count = self.cache.batch_set(translation_dict)
            
            self.logger.debug(f"Batch cached {count} translations from {source_language} to {target_language}")
            
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cache usage.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            cache_stats = self.cache.stats()
            
            # Add manager-level stats
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0
            
            stats = {
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'total_requests': total_requests,
                'hit_rate': hit_rate,
                'cache_type': self.cache_type,
                **cache_stats
            }
            
            return stats
    
    def cleanup(self) -> None:
        """
        Perform cache maintenance operations.
        """
        with self.lock:
            removed = self.cache.cleanup()
            self.logger.info(f"Cache cleanup removed {removed} expired entries")
    
    def flush(self) -> None:
        """
        Ensure all pending writes are committed to persistent storage.
        """
        with self.lock:
            success = self.cache.flush()
            if success:
                self.logger.debug("Cache flushed successfully")
            else:
                self.logger.warning("Failed to flush cache")