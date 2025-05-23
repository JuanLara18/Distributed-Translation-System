#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Spark configuration utilities with automatic resource detection.
Add this to modules/utilities.py or create a new file modules/spark_optimizer.py
"""

import os
import psutil
import logging
from typing import Dict, Any, Optional, Tuple
from pyspark.sql import SparkSession


def detect_system_resources() -> Dict[str, Any]:
    """
    Automatically detect system resources for optimal Spark configuration.
    
    Returns:
        Dictionary with detected system resources
    """
    # Get CPU information
    cpu_count = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    
    # Get memory information
    memory = psutil.virtual_memory()
    total_memory_gb = memory.total / (1024**3)
    available_memory_gb = memory.available / (1024**3)
    
    return {
        'cpu_logical': cpu_count,
        'cpu_physical': cpu_physical,
        'total_memory_gb': total_memory_gb,
        'available_memory_gb': available_memory_gb,
        'memory_usage_percent': memory.percent
    }


def calculate_optimal_spark_config(system_resources: Dict[str, Any], 
                                   workload_type: str = "medium") -> Dict[str, Any]:
    """
    Calculate optimal Spark configuration based on system resources.
    
    Args:
        system_resources: System resource information
        workload_type: "light", "medium", "heavy" - affects memory allocation
        
    Returns:
        Optimized Spark configuration
    """
    cpu_count = system_resources['cpu_logical']
    available_memory_gb = system_resources['available_memory_gb']
    
    # Reserve memory for OS and other processes
    if workload_type == "light":
        memory_reservation_gb = max(4, available_memory_gb * 0.1)  # Reserve 10% or 4GB
        concurrency_factor = 1.5
    elif workload_type == "medium":
        memory_reservation_gb = max(8, available_memory_gb * 0.15)  # Reserve 15% or 8GB
        concurrency_factor = 2.0
    else:  # heavy
        memory_reservation_gb = max(16, available_memory_gb * 0.2)  # Reserve 20% or 16GB
        concurrency_factor = 3.0
    
    usable_memory_gb = available_memory_gb - memory_reservation_gb
    
    # Calculate optimal executor configuration
    # Use 80% of CPUs for executors, leaving some for driver and OS
    executor_cores = min(6, max(2, int(cpu_count * 0.8)))  # Cap at 6 cores per executor
    num_executors = max(1, cpu_count // executor_cores)
    
    # Calculate memory per executor
    executor_memory_gb = max(2, int((usable_memory_gb * 0.7) / num_executors))
    driver_memory_gb = max(2, min(executor_memory_gb, usable_memory_gb * 0.2))
    
    # Calculate parallelism
    default_parallelism = int(cpu_count * concurrency_factor)
    
    return {
        'executor_memory': f"{executor_memory_gb}g",
        'driver_memory': f"{driver_memory_gb}g",
        'executor_cores': executor_cores,
        'default_parallelism': default_parallelism,
        'num_executors': num_executors,
        'max_result_size': f"{min(4, driver_memory_gb // 2)}g",
        # Additional optimizations
        'executor_memory_overhead': f"{max(1, executor_memory_gb // 10)}g",
        'driver_memory_overhead': f"{max(1, driver_memory_gb // 10)}g"
    }


def create_optimized_spark_session(app_name: str = "distributed_translation",
                                   config: Optional[Dict[str, Any]] = None,
                                   auto_optimize: bool = True,
                                   workload_type: str = "medium") -> SparkSession:
    """
    Create and configure a Spark session with automatic optimization.
    
    Args:
        app_name: Name of the Spark application
        config: Manual configuration dictionary (overrides auto-optimization)
        auto_optimize: Whether to automatically detect and optimize configuration
        workload_type: "light", "medium", "heavy" for memory allocation strategy
        
    Returns:
        Optimized SparkSession
    """
    logger = logging.getLogger(__name__)
    
    if auto_optimize and not config:
        logger.info("Auto-detecting system resources for Spark optimization...")
        system_resources = detect_system_resources()
        
        logger.info(f"Detected: {system_resources['cpu_logical']} CPUs, "
                   f"{system_resources['total_memory_gb']:.1f}GB total RAM, "
                   f"{system_resources['available_memory_gb']:.1f}GB available")
        
        spark_config = calculate_optimal_spark_config(system_resources, workload_type)
        logger.info(f"Calculated optimal Spark config: {spark_config}")
    else:
        # Use provided config or conservative defaults
        spark_config = config or {
            "executor_memory": "4g",
            "driver_memory": "4g",
            "executor_cores": 2,
            "default_parallelism": 4
        }
    
    # Build SparkSession with optimized configuration
    builder = SparkSession.builder.appName(app_name)
    
    # Core Spark settings
    builder = builder.config("spark.executor.memory", spark_config["executor_memory"])
    builder = builder.config("spark.driver.memory", spark_config["driver_memory"])
    builder = builder.config("spark.executor.cores", spark_config["executor_cores"])
    builder = builder.config("spark.default.parallelism", spark_config["default_parallelism"])
    
    # Memory overhead settings (important for large datasets)
    if "executor_memory_overhead" in spark_config:
        builder = builder.config("spark.executor.memoryOverhead", spark_config["executor_memory_overhead"])
    if "driver_memory_overhead" in spark_config:
        builder = builder.config("spark.driver.memoryOverhead", spark_config["driver_memory_overhead"])
    
    # Result size limit
    if "max_result_size" in spark_config:
        builder = builder.config("spark.driver.maxResultSize", spark_config["max_result_size"])
    
    # Performance optimizations
    builder = builder.config("spark.sql.adaptive.enabled", "true")
    builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    builder = builder.config("spark.sql.adaptive.skewJoin.enabled", "true")
    builder = builder.config("spark.sql.shuffle.partitions", str(spark_config["default_parallelism"] * 2))
    
    # Garbage collection optimizations
    builder = builder.config("spark.driver.extraJavaOptions", 
                            "-XX:+UseG1GC -XX:+UnlockDiagnosticVMOptions -XX:+G1PrintRegionRememberedSetInfo")
    builder = builder.config("spark.executor.extraJavaOptions", 
                            "-XX:+UseG1GC -XX:+UnlockDiagnosticVMOptions -XX:+G1PrintRegionRememberedSetInfo")
    
    # Serialization optimization
    builder = builder.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    
    # Network optimizations for large datasets
    builder = builder.config("spark.network.timeout", "600s")
    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    
    # Create the session
    spark = builder.getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("ERROR")
    
    logger.info(f"Created optimized Spark session: {app_name}")
    logger.info(f"Executor memory: {spark_config['executor_memory']}, "
               f"Driver memory: {spark_config['driver_memory']}, "
               f"Cores per executor: {spark_config['executor_cores']}, "
               f"Parallelism: {spark_config['default_parallelism']}")
    
    return spark


def get_recommended_config_for_system() -> Dict[str, Any]:
    """
    Get recommended configuration values for the current system.
    
    Returns:
        Dictionary with recommended configuration for config.yaml
    """
    system_resources = detect_system_resources()
    
    # Calculate configurations for different workload types
    configs = {}
    for workload in ["light", "medium", "heavy"]:
        spark_config = calculate_optimal_spark_config(system_resources, workload)
        configs[workload] = {
            'spark': spark_config,
            'batch_size': {
                'light': min(20, system_resources['cpu_logical'] * 2),
                'medium': min(50, system_resources['cpu_logical'] * 4),
                'heavy': min(100, system_resources['cpu_logical'] * 8)
            }[workload]
        }
    
    return {
        'system_info': system_resources,
        'recommendations': configs
    }


def override_java_options_for_spark():
    """
    Override restrictive Java options that might limit Spark performance.
    Call this before creating the Spark session.
    """
    logger = logging.getLogger(__name__)
    
    # Detect current Java options
    current_java_options = os.environ.get('_JAVA_OPTIONS', '')
    if current_java_options:
        logger.warning(f"Current _JAVA_OPTIONS detected: {current_java_options}")
        logger.warning("These options may limit Spark performance. Consider unsetting _JAVA_OPTIONS.")
    
    # Get system memory for reasonable defaults
    system_resources = detect_system_resources()
    available_memory_gb = system_resources['available_memory_gb']
    
    # Calculate reasonable JVM settings
    max_heap_gb = min(32, int(available_memory_gb * 0.4))  # Don't exceed 32GB for JVM efficiency
    
    # Set optimized Java options for Spark
    optimized_options = [
        f"-Xmx{max_heap_gb}g",
        f"-Xms{max_heap_gb//2}g",
        "-XX:+UseG1GC",
        "-XX:MaxGCPauseMillis=200",
        "-XX:+UnlockDiagnosticVMOptions",
        "-XX:+G1PrintRegionRememberedSetInfo",
        "-XX:MaxMetaspaceSize=512m",
        "-XX:CompressedClassSpaceSize=256m"
    ]
    
    # Only override if we're not in a restricted environment
    if '_JAVA_OPTIONS' in os.environ:
        logger.info("Detected restrictive _JAVA_OPTIONS. You may want to run:")
        logger.info("unset _JAVA_OPTIONS")
        logger.info("before running this script for better performance.")
    
    return " ".join(optimized_options)


def print_system_optimization_report():
    """
    Print a detailed report of system resources and optimization recommendations.
    """
    recommendations = get_recommended_config_for_system()
    system_info = recommendations['system_info']
    
    print("=" * 80)
    print("SYSTEM OPTIMIZATION REPORT")
    print("=" * 80)
    
    print(f"\nSYSTEM RESOURCES:")
    print(f"  CPUs (logical): {system_info['cpu_logical']}")
    print(f"  CPUs (physical): {system_info['cpu_physical']}")
    print(f"  Total RAM: {system_info['total_memory_gb']:.1f} GB")
    print(f"  Available RAM: {system_info['available_memory_gb']:.1f} GB")
    print(f"  Memory usage: {system_info['memory_usage_percent']:.1f}%")
    
    print(f"\nRECOMMENDED CONFIGURATIONS:")
    
    for workload_type, config in recommendations['recommendations'].items():
        print(f"\n  {workload_type.upper()} WORKLOAD:")
        spark_config = config['spark']
        print(f"    spark:")
        print(f"      executor_memory: \"{spark_config['executor_memory']}\"")
        print(f"      driver_memory: \"{spark_config['driver_memory']}\"")
        print(f"      executor_cores: {spark_config['executor_cores']}")
        print(f"      default_parallelism: {spark_config['default_parallelism']}")
        print(f"    batch_size: {config['batch_size']}")
    
    # Check for Java options issues
    java_options = os.environ.get('_JAVA_OPTIONS', '')
    if java_options:
        print(f"\n⚠️  WARNING: Restrictive _JAVA_OPTIONS detected:")
        print(f"    {java_options}")
        print(f"    Consider running: unset _JAVA_OPTIONS")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run this script directly to see optimization recommendations
    print_system_optimization_report()