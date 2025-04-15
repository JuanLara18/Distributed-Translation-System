import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import json
import yaml
import time
import psutil
import logging
from datetime import datetime
import sqlite3
import glob
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple, Optional

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from config import ConfigManager
from modules.cache import CacheManager
from modules.checkpoint import CheckpointManager
from modules.utilities import set_up_logging, format_file_size, format_stats_report

# Set up logger
logger = set_up_logging(level="INFO", log_file="streamlit_app.log")

# Page configuration
st.set_page_config(
    page_title="Distributed Translation System",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file"""
    if not os.path.exists(config_path):
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {}

def save_config(config: dict, config_path: str) -> bool:
    """Save configuration to a YAML file"""
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")
        return False

def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "cpu": {
            "percent": cpu_percent,
            "count": psutil.cpu_count(),
            "cores": psutil.cpu_count(logical=False)
        },
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        },
        "disk": {
            "total": disk.total,
            "free": disk.free,
            "used": disk.used,
            "percent": disk.percent
        }
    }

def explore_database(db_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Explore SQLite database and return statistics
    """
    if not os.path.exists(db_path):
        return pd.DataFrame(), {}
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get translations
        translations_df = pd.read_sql(
            "SELECT source_language, target_language, timestamp FROM translations LIMIT 1000", 
            conn
        )
        
        # Get statistics
        stats = {}
        
        # Total entries
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM translations")
        stats['total_entries'] = cur.fetchone()[0]
        
        # Languages
        cur.execute("SELECT DISTINCT source_language FROM translations")
        stats['source_languages'] = [row[0] for row in cur.fetchall()]
        
        cur.execute("SELECT DISTINCT target_language FROM translations")
        stats['target_languages'] = [row[0] for row in cur.fetchall()]
        
        # Size of database
        stats['db_size'] = os.path.getsize(db_path)
        
        # Entries by language pair
        language_pairs_df = pd.read_sql(
            """
            SELECT source_language, target_language, COUNT(*) as count 
            FROM translations 
            GROUP BY source_language, target_language
            """, 
            conn
        )
        stats['language_pairs'] = language_pairs_df.to_dict('records')
        
        # Entries over time
        time_data = pd.read_sql(
            """
            SELECT 
                datetime(timestamp, 'unixepoch') as date,
                COUNT(*) as count
            FROM translations
            GROUP BY date
            ORDER BY date
            """, 
            conn
        )
        stats['time_data'] = time_data.to_dict('records')
        
        conn.close()
        return translations_df, stats
    
    except Exception as e:
        st.error(f"Error exploring database: {str(e)}")
        return pd.DataFrame(), {}

def get_checkpoints_info(checkpoint_dir: str) -> Dict[str, Any]:
    """Get information about available checkpoints"""
    if not os.path.exists(checkpoint_dir):
        return {}
    
    try:
        # Find checkpoint files
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.parquet'))
        
        # Check metadata file
        metadata_path = os.path.join(checkpoint_dir, 'checkpoint_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Global state
        global_state_path = os.path.join(checkpoint_dir, 'global_state.json')
        if os.path.exists(global_state_path):
            with open(global_state_path, 'r') as f:
                global_state = json.load(f)
        else:
            global_state = {}
        
        return {
            "checkpoint_files": checkpoint_files,
            "checkpoint_count": len(checkpoint_files),
            "metadata": metadata,
            "global_state": global_state
        }
    except Exception as e:
        st.error(f"Error getting checkpoint info: {str(e)}")
        return {}

def run_translation_job(config_path: str) -> bool:
    """Run a translation job with the specified configuration"""
    try:
        # We'll use the subprocess module to run the job in a separate process
        import subprocess
        
        # Display spinner while job runs
        with st.spinner('Running translation job...'):
            # Build command
            cmd = [sys.executable, "main.py", "--config", config_path]
            
            # Add options based on form inputs
            if st.session_state.get('verbose', False):
                cmd.append("--verbose")
            
            if st.session_state.get('resume', False):
                cmd.append("--resume")
            
            if st.session_state.get('force_restart', False):
                cmd.append("--force_restart")
            
            # Run command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Create output placeholders
            output_placeholder = st.empty()
            error_placeholder = st.empty()
            
            # Show output in real-time
            stdout_lines = []
            stderr_lines = []
            
            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()
                
                if stdout_line:
                    stdout_lines.append(stdout_line)
                    output_placeholder.text('\n'.join(stdout_lines[-20:]))  # Show last 20 lines
                
                if stderr_line:
                    stderr_lines.append(stderr_line)
                    error_placeholder.text('\n'.join(stderr_lines[-20:]))  # Show last 20 lines
                
                # Check if process is still running
                if process.poll() is not None:
                    # Get any remaining output
                    stdout_lines.extend(process.stdout.readlines())
                    stderr_lines.extend(process.stderr.readlines())
                    output_placeholder.text('\n'.join(stdout_lines[-20:]))
                    error_placeholder.text('\n'.join(stderr_lines[-20:]))
                    break
            
            # Check return code
            return_code = process.poll()
            
            if return_code == 0:
                st.success("Translation job completed successfully!")
                return True
            else:
                st.error(f"Translation job failed with exit code {return_code}")
                return False
    
    except Exception as e:
        st.error(f"Error running translation job: {str(e)}")
        return False

# Sidebar navigation
st.sidebar.title("Distributed Translation System")
page = st.sidebar.selectbox(
    "Navigation", 
    ["Dashboard", "Configuration", "Job Control", "Cache Explorer", "Checkpoints", "About"]
)

# System resources in sidebar
st.sidebar.header("System Resources")
resources = get_system_resources()

# CPU usage
cpu_col1, cpu_col2 = st.sidebar.columns(2)
cpu_col1.metric("CPU Usage", f"{resources['cpu']['percent']}%")
cpu_col2.metric("CPU Cores", f"{resources['cpu']['cores']} ({resources['cpu']['count']} threads)")

# Memory usage
mem_col1, mem_col2 = st.sidebar.columns(2)
mem_col1.metric("Memory Usage", f"{resources['memory']['percent']}%")
mem_col2.metric("Available Memory", format_file_size(resources['memory']['available']))

# Disk usage
disk_col1, disk_col2 = st.sidebar.columns(2)
disk_col1.metric("Disk Usage", f"{resources['disk']['percent']}%")
disk_col2.metric("Free Space", format_file_size(resources['disk']['free']))

# Default configuration path
default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

# Dashboard page
if page == "Dashboard":
    st.title("📊 Dashboard")
    
    # Overview section
    st.header("System Overview")
    
    # Check if configuration exists
    if os.path.exists(default_config_path):
        config = load_config(default_config_path)
        
        # Load cache statistics if available
        cache_path = config.get('cache', {}).get('location', './cache/translations.db')
        if os.path.exists(cache_path):
            translations_df, cache_stats = explore_database(cache_path)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Translations", f"{cache_stats.get('total_entries', 0):,}")
            col2.metric("Source Languages", len(cache_stats.get('source_languages', [])))
            col3.metric("Target Languages", len(cache_stats.get('target_languages', [])))
            col4.metric("Cache Size", format_file_size(cache_stats.get('db_size', 0)))
            
            # Display charts
            st.subheader("Translation Distribution")
            
            if cache_stats.get('language_pairs'):
                lang_pairs_df = pd.DataFrame(cache_stats['language_pairs'])
                
                # Create chart
                fig = px.bar(
                    lang_pairs_df,
                    x='source_language',
                    y='count',
                    color='target_language',
                    title="Translations by Language Pair",
                    labels={'count': 'Number of Translations', 'source_language': 'Source Language', 'target_language': 'Target Language'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Timeline chart
            if cache_stats.get('time_data'):
                time_df = pd.DataFrame(cache_stats['time_data'])
                time_df['date'] = pd.to_datetime(time_df['date'])
                
                fig = px.line(
                    time_df,
                    x='date',
                    y='count',
                    title="Translations Over Time",
                    labels={'count': 'Number of Translations', 'date': 'Date'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Cache database not found at {cache_path}. Run a translation job to generate statistics.")
    else:
        st.info("No configuration found. Go to the Configuration page to set up the system.")
    
    # Recent jobs section
    st.header("Recent Jobs")
    
    # Check for job statistics
    stats_file = "translation_stats.json"
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f:
                job_stats = json.load(f)
            
            # Display job stats
            st.subheader("Latest Job Statistics")
            
            # Basic stats
            cols = st.columns(4)
            cols[0].metric("Total Rows", f"{job_stats.get('total_rows', 0):,}")
            cols[1].metric("Translated Rows", f"{job_stats.get('translated_rows', 0):,}")
            cols[2].metric("API Calls", f"{job_stats.get('api_calls', 0):,}")
            cols[3].metric("Cache Hits", f"{job_stats.get('cached_hits', 0):,}")
            
            # Calculate cache hit rate
            if 'api_calls' in job_stats and 'cached_hits' in job_stats:
                total = job_stats['api_calls'] + job_stats['cached_hits']
                if total > 0:
                    hit_rate = (job_stats['cached_hits'] / total) * 100
                    st.metric("Cache Hit Rate", f"{hit_rate:.2f}%")
            
            # Show full stats report
            with st.expander("View Detailed Statistics"):
                if 'start_time' in job_stats and 'end_time' in job_stats:
                    st.text(format_stats_report(job_stats))
                else:
                    st.json(job_stats)
                
        except Exception as e:
            st.error(f"Error loading job statistics: {str(e)}")
    else:
        st.info("No job statistics found. Run a translation job to generate statistics.")

# Configuration page
elif page == "Configuration":
    st.title("⚙️ Configuration")
    
    # Load existing configuration if available
    config_path = st.text_input("Configuration Path", value=default_config_path)
    
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        # Start with default configuration
        config = {
            'input_file': None,
            'output_file': None,
            'columns_to_translate': [],
            'source_language_column': None,
            'target_language': 'english',
            'batch_size': 10,
            'openai': {
                'model': 'gpt-3.5-turbo',
                'temperature': 0.1,
                'max_tokens': 1500,
                'api_key_env': 'OPENAI_API_KEY'
            },
            'checkpoint': {
                'enabled': True,
                'interval': 1,
                'directory': './checkpoints',
                'max_checkpoints': 5
            },
            'cache': {
                'type': 'sqlite',
                'location': './cache/translations.db',
                'ttl': 2592000,  # 30 days in seconds
            },
            'spark': {
                'executor_memory': '4g',
                'driver_memory': '4g',
                'executor_cores': 2,
                'default_parallelism': 4
            },
            'logging': {
                'level': 'INFO',
                'log_file': 'translation_process.log'
            }
        }
    
    with st.form("config_form"):
        st.header("Basic Settings")
        
        # File paths
        col1, col2 = st.columns(2)
        config['input_file'] = col1.text_input("Input File", value=config.get('input_file', ''))
        config['output_file'] = col2.text_input("Output File", value=config.get('output_file', ''))
        
        # Translation columns
        columns_str = ','.join(config.get('columns_to_translate', []))
        columns_input = st.text_input("Columns to Translate (comma-separated)", value=columns_str)
        config['columns_to_translate'] = [col.strip() for col in columns_input.split(',') if col.strip()]
        
        col1, col2 = st.columns(2)
        config['source_language_column'] = col1.text_input(
            "Source Language Column (leave empty for auto-detection)", 
            value=config.get('source_language_column', '')
        )
        config['target_language'] = col2.text_input("Target Language", value=config.get('target_language', 'english'))
        
        # Advanced settings in expandable sections
        with st.expander("OpenAI Settings"):
            openai_config = config.get('openai', {})
            col1, col2 = st.columns(2)
            openai_config['model'] = col1.selectbox(
                "Model", 
                options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo-instruct"],
                index=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo-instruct"].index(openai_config.get('model', 'gpt-3.5-turbo'))
            )
            openai_config['temperature'] = col2.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=float(openai_config.get('temperature', 0.1)),
                step=0.1
            )
            openai_config['max_tokens'] = col1.number_input(
                "Max Tokens", 
                min_value=1, 
                max_value=8192, 
                value=int(openai_config.get('max_tokens', 1500))
            )
            openai_config['api_key_env'] = col2.text_input(
                "API Key Environment Variable", 
                value=openai_config.get('api_key_env', 'OPENAI_API_KEY')
            )
            
            # Update the config
            config['openai'] = openai_config
        
        with st.expander("Cache Settings"):
            cache_config = config.get('cache', {})
            col1, col2 = st.columns(2)
            cache_config['type'] = col1.selectbox(
                "Cache Type", 
                options=["sqlite", "postgres", "memory"],
                index=["sqlite", "postgres", "memory"].index(cache_config.get('type', 'sqlite'))
            )
            cache_config['location'] = col2.text_input(
                "Cache Location", 
                value=cache_config.get('location', './cache/translations.db')
            )
            
            # Show connection string for postgres
            if cache_config['type'] == 'postgres':
                cache_config['connection_string'] = st.text_input(
                    "PostgreSQL Connection String", 
                    value=cache_config.get('connection_string', 'postgresql://user:password@localhost:5432/translations')
                )
            
            # Cache TTL
            ttl_days = cache_config.get('ttl', 2592000) / 86400  # Convert seconds to days
            cache_config['ttl'] = st.slider(
                "Cache TTL (days)", 
                min_value=1, 
                max_value=365, 
                value=int(ttl_days)
            ) * 86400  # Convert days back to seconds
            
            # Update the config
            config['cache'] = cache_config
            
        with st.expander("Checkpoint Settings"):
            checkpoint_config = config.get('checkpoint', {})
            col1, col2 = st.columns(2)
            checkpoint_config['enabled'] = col1.checkbox(
                "Enable Checkpointing", 
                value=checkpoint_config.get('enabled', True)
            )
            checkpoint_config['interval'] = col2.number_input(
                "Checkpoint Interval (partitions)", 
                min_value=1, 
                max_value=100, 
                value=int(checkpoint_config.get('interval', 1))
            )
            checkpoint_config['directory'] = col1.text_input(
                "Checkpoint Directory", 
                value=checkpoint_config.get('directory', './checkpoints')
            )
            checkpoint_config['max_checkpoints'] = col2.number_input(
                "Max Checkpoints to Keep", 
                min_value=1, 
                max_value=100, 
                value=int(checkpoint_config.get('max_checkpoints', 5))
            )
            
            # Update the config
            config['checkpoint'] = checkpoint_config
            
        with st.expander("Spark Settings"):
            spark_config = config.get('spark', {})
            col1, col2 = st.columns(2)
            spark_config['executor_memory'] = col1.text_input(
                "Executor Memory", 
                value=spark_config.get('executor_memory', '4g')
            )
            spark_config['driver_memory'] = col2.text_input(
                "Driver Memory", 
                value=spark_config.get('driver_memory', '4g')
            )
            spark_config['executor_cores'] = col1.number_input(
                "Executor Cores", 
                min_value=1, 
                max_value=16, 
                value=int(spark_config.get('executor_cores', 2))
            )
            spark_config['default_parallelism'] = col2.number_input(
                "Default Parallelism", 
                min_value=1, 
                max_value=32, 
                value=int(spark_config.get('default_parallelism', 4))
            )
            
            # Update the config
            config['spark'] = spark_config
            
        with st.expander("Logging Settings"):
            logging_config = config.get('logging', {})
            col1, col2 = st.columns(2)
            logging_config['level'] = col1.selectbox(
                "Log Level", 
                options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(logging_config.get('level', 'INFO'))
            )
            logging_config['log_file'] = col2.text_input(
                "Log File", 
                value=logging_config.get('log_file', 'translation_process.log')
            )
            
            # Update the config
            config['logging'] = logging_config
            
        # Save button
        submitted = st.form_submit_button("Save Configuration")
        
        if submitted:
            # Save the configuration
            if save_config(config, config_path):
                st.success(f"Configuration saved to {config_path}")
            else:
                st.error("Failed to save configuration")

# Job Control page
elif page == "Job Control":
    st.title("🚀 Job Control")
    
    # Check if configuration exists
    if not os.path.exists(default_config_path):
        st.warning("No configuration found. Please create a configuration first.")
        st.button("Go to Configuration", on_click=lambda: st.session_state.update({"page": "Configuration"}))
    else:
        # Load the configuration
        config = load_config(default_config_path)
        
        # Display job parameters
        with st.form("job_form"):
            st.header("Job Parameters")
            
            # Basic info
            col1, col2 = st.columns(2)
            col1.text(f"Input File: {config.get('input_file', 'Not set')}")
            col2.text(f"Output File: {config.get('output_file', 'Not set')}")
            
            st.text(f"Columns to Translate: {', '.join(config.get('columns_to_translate', []))}")
            st.text(f"Target Language: {config.get('target_language', 'Not set')}")
            
            # Job options
            st.subheader("Options")
            col1, col2, col3 = st.columns(3)
            st.session_state['verbose'] = col1.checkbox("Verbose Logging", value=False)
            st.session_state['resume'] = col2.checkbox("Resume from Checkpoint", value=True)
            st.session_state['force_restart'] = col3.checkbox("Force Restart", value=False)
            
            # Submit button
            submitted = st.form_submit_button("Start Translation Job")
            
            if submitted:
                # Run the job
                success = run_translation_job(default_config_path)
                
                if success:
                    # Provide a link to view results
                    st.success("Job completed successfully!")
                    
                    # If we have an output file, provide a link to download it
                    output_file = config.get('output_file')
                    if output_file and os.path.exists(output_file):
                        with open(output_file, 'rb') as f:
                            st.download_button(
                                label="Download Results",
                                data=f,
                                file_name=os.path.basename(output_file),
                                mime="application/octet-stream"
                            )
                else:
                    st.error("Job failed. Check the logs for details.")
        
        # Job history section
        st.header("Job History")
        
        # Check for log file
        log_file = config.get('logging', {}).get('log_file', 'translation_process.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.readlines()
            
            # Find job runs in the logs
            job_runs = []
            current_job = None
            
            for line in log_content:
                if "Initializing translation process" in line:
                    # Start of a new job
                    timestamp = line.split(" - ")[0]
                    current_job = {"start": timestamp, "end": None, "status": "Running"}
                elif "Processing completed successfully" in line and current_job:
                    # Job completed successfully
                    timestamp = line.split(" - ")[0]
                    current_job["end"] = timestamp
                    current_job["status"] = "Completed"
                    job_runs.append(current_job)
                    current_job = None
                elif "Processing failed" in line and current_job:
                    # Job failed
                    timestamp = line.split(" - ")[0]
                    current_job["end"] = timestamp
                    current_job["status"] = "Failed"
                    job_runs.append(current_job)
                    current_job = None
            
            # Show job runs
            if job_runs:
                job_df = pd.DataFrame(job_runs)
                st.dataframe(job_df)
            else:
                st.info("No job runs found in the logs.")
        else:
            st.info("No log file found.")

# Cache Explorer page
elif page == "Cache Explorer":
    st.title("🔍 Cache Explorer")
    
    # Load the configuration to get cache path
    if os.path.exists(default_config_path):
        config = load_config(default_config_path)
        cache_path = config.get('cache', {}).get('location', './cache/translations.db')
        
        if os.path.exists(cache_path):
            st.header("Cache Database")
            
            # Get cache data
            translations_df, cache_stats = explore_database(cache_path)
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Entries", f"{cache_stats.get('total_entries', 0):,}")
            col2.metric("Source Languages", len(cache_stats.get('source_languages', [])))
            col3.metric("Target Languages", len(cache_stats.get('target_languages', [])))
            col4.metric("Database Size", format_file_size(cache_stats.get('db_size', 0)))
            
            # Display source and target languages
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Source Languages")
                st.write(", ".join(sorted(cache_stats.get('source_languages', []))))
            
            with col2:
                st.subheader("Target Languages")
                st.write(", ".join(sorted(cache_stats.get('target_languages', []))))
            
            # Display language pair distribution
            if cache_stats.get('language_pairs'):
                st.subheader("Language Pair Distribution")
                pairs_df = pd.DataFrame(cache_stats['language_pairs'])
                st.dataframe(pairs_df)
                
                # Visualization
                fig = px.pie(
                    pairs_df, 
                    values='count', 
                    names='source_language',
                    title="Translation Distribution by Source Language"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Sample entries
            if not translations_df.empty:
                st.subheader("Sample Entries")
                st.dataframe(translations_df)
            
            # Cache management actions
            st.header("Cache Management")
            col1, col2 = st.columns(2)
            
            if col1.button("Clean Expired Entries"):
                try:
                    # Create CacheManager
                    cache_config = config.get('cache', {})
                    cache_manager = CacheManager(config)
                    
                    # Clean up expired entries
                    removed = cache_manager.cache.cleanup()
                    st.success(f"Removed {removed} expired entries from cache")
                except Exception as e:
                    st.error(f"Error cleaning cache: {str(e)}")
            
            if col2.button("Clear Cache"):
                # Add a confirmation dialog
                confirm = st.checkbox("I understand this will delete all cached translations")
                
                if confirm:
                    try:
                        # Create CacheManager
                        cache_manager = CacheManager(config)
                        
                        # Clear cache
                        success = cache_manager.cache.clear()
                        
                        if success:
                            st.success("Cache cleared successfully")
                        else:
                            st.error("Failed to clear cache")
                    except Exception as e:
                        st.error(f"Error clearing cache: {str(e)}")
        else:
            st.info(f"Cache database not found at {cache_path}. Run a translation job to generate cache.")
    else:
        st.warning("No configuration found. Please create a configuration first.")

# Checkpoints page
elif page == "Checkpoints":
    st.title("🔄 Checkpoints")
    
    # Load the configuration to get checkpoint path
    if os.path.exists(default_config_path):
        config = load_config(default_config_path)
        checkpoint_dir = config.get('checkpoint', {}).get('directory', './checkpoints')
        
        if os.path.exists(checkpoint_dir):
            # Get checkpoint info
            checkpoint_info = get_checkpoints_info(checkpoint_dir)
            
            # Display checkpoint statistics
            st.header("Checkpoint Information")
            
            col1, col2 = st.columns(2)
            col1.metric("Number of Checkpoints", checkpoint_info.get('checkpoint_count', 0))
            
            # Check if we have metadata
            metadata = checkpoint_info.get('metadata', {})
            if metadata:
                col2.metric("Processed Partitions", len(metadata.get('processed_partitions', [])))
                
                # Show processed partitions
                st.subheader("Processed Partitions")
                st.write(", ".join(map(str, sorted(metadata.get('processed_partitions', [])))))
            
            # Show global state if available
            global_state = checkpoint_info.get('global_state', {})
            if global_state:
                st.subheader("Global State")
                
                # Format timestamp
                if 'timestamp' in global_state:
                    timestamp = datetime.fromtimestamp(global_state['timestamp'])
                    global_state['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                
                # Show statistics
                cols = st.columns(4)
                if 'total_rows' in global_state:
                    cols[0].metric("Total Rows", f"{global_state['total_rows']:,}")
                if 'translated_rows' in global_state:
                    cols[1].metric("Translated Rows", f"{global_state['translated_rows']:,}")
                if 'api_calls' in global_state:
                    cols[2].metric("API Calls", f"{global_state['api_calls']:,}")
                if 'cached_hits' in global_state:
                    cols[3].metric("Cache Hits", f"{global_state['cached_hits']:,}")
                
                # Show full global state
                with st.expander("View Complete Global State"):
                    st.json(global_state)
            
            # Checkpoint file list
            if checkpoint_info.get('checkpoint_files'):
                st.subheader("Checkpoint Files")
                
                for file in sorted(checkpoint_info['checkpoint_files']):
                    file_info = {
                        'filename': os.path.basename(file),
                        'size': format_file_size(os.path.getsize(file)),
                        'modified': datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')
                    }
                    st.text(f"{file_info['filename']} - Size: {file_info['size']} - Modified: {file_info['modified']}")
            
            # Checkpoint management
            st.header("Checkpoint Management")
            col1, col2 = st.columns(2)
            
            if col1.button("Clear All Checkpoints"):
                # Add a confirmation dialog
                confirm = st.checkbox("I understand this will delete all checkpoints")
                
                if confirm:
                    try:
                        # Create CheckpointManager
                        checkpoint_manager = CheckpointManager(config)
                        
                        # Clear checkpoints
                        success = checkpoint_manager.clear_checkpoints()
                        
                        if success:
                            st.success("Checkpoints cleared successfully")
                        else:
                            st.error("Failed to clear checkpoints")
                    except Exception as e:
                        st.error(f"Error clearing checkpoints: {str(e)}")
            
            if 'global_state.json' in [os.path.basename(f) for f in checkpoint_info.get('checkpoint_files', [])]:
                if col2.button("Download Global State"):
                    global_state_path = os.path.join(checkpoint_dir, 'global_state.json')
                    with open(global_state_path, 'r') as f:
                        st.download_button(
                            label="Download Global State JSON",
                            data=f,
                            file_name="global_state.json",
                            mime="application/json"
                        )
        else:
            st.info(f"Checkpoint directory not found at {checkpoint_dir}. Run a translation job to generate checkpoints.")
    else:
        st.warning("No configuration found. Please create a configuration first.")

# About page
elif page == "About":
    st.title("ℹ️ About Distributed Translation System")
    
    st.markdown("""
    ## What It Does

    This system allows you to translate text columns in data files (Stata, CSV, Parquet, JSON) with these key features:

    - **Distributed processing** with PySpark for handling large datasets efficiently
    - **Smart caching** to avoid redundant API calls and reduce costs
    - **Fault tolerance** with checkpointing to resume interrupted processes
    - **Multiple file formats** with preserved metadata (especially for Stata files)
    - **Language detection** for automatic source language identification
    - **Batch processing** for optimized throughput

    ## Architecture

    The system uses a modular architecture with well-defined interfaces:

    - **TranslationOrchestrator** (in `main.py`): Central controller coordinating all components
    - **ConfigManager** (in `config.py`): Manages configuration loading and validation
    - **TranslationManager** (in `modules/translator.py`): Manages the translation process
    - **CacheManager** (in `modules/cache.py`): Coordinates caching operations
    - **CheckpointManager** (in `modules/checkpoint.py`): Handles state persistence
    - **DataReader/DataWriter** (in `modules/file_manager.py`): Handle I/O operations
    
    ## Future Improvements
    
    - Support for additional translation models and APIs (DeepL, Google Translate, etc.)
    - Advanced distributed architecture using Kubernetes
    - Interactive translation quality metrics and evaluation
    - Support for more specialized file formats and content types
    - Custom terminology management and glossaries
    """)
    
    # System information
    st.header("System Information")
    
    col1, col2 = st.columns(2)
    col1.metric("Python Version", sys.version.split()[0])
    col1.metric("Operating System", os.name)
    
    # Check if required packages are installed
    required_packages = ["pyspark", "pandas", "openai", "pyreadstat", "sqlalchemy", "plotly"]
    
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "Unknown")
            col2.text(f"✅ {package}: {version}")
        except ImportError:
            col2.text(f"❌ {package}: Not installed")
    
    # API key status
    st.header("OpenAI API Status")
    
    api_key_env = config.get('openai', {}).get('api_key_env', 'OPENAI_API_KEY')
    if os.environ.get(api_key_env):
        st.success(f"OpenAI API key found in environment variable: {api_key_env}")
    else:
        st.error(f"OpenAI API key not found in environment variable: {api_key_env}")
        
        # Show instructions for setting API key
        with st.expander("How to Set API Key"):
            st.markdown("""
            ### Setting the OpenAI API Key
            
            You need to set the OpenAI API key as an environment variable before running the app:
            
            #### Linux/Mac:
            ```bash
            export OPENAI_API_KEY=your-api-key-here
            ```
            
            #### Windows:
            ```
            set OPENAI_API_KEY=your-api-key-here
            ```
            
            ### Permanent Solution
            
            For a more permanent solution, add the API key to your environment variables through your operating system settings.
            """)

# Main function to run the app
if __name__ == "__main__":
    try:
        # Check if this is the first run
        if "initialized" not in st.session_state:
            # Set initial page if needed
            if not os.path.exists(default_config_path):
                st.session_state["page"] = "Configuration"
            
            # Mark as initialized
            st.session_state["initialized"] = True
    except Exception as e:
        st.error(f"Error initializing app: {str(e)}")