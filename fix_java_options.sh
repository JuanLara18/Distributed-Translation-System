#!/bin/bash
# fix_java_options.sh
# Script to optimize Java options for Spark on your 128GB/12CPU system

echo "Current Java options:"
echo "_JAVA_OPTIONS: $JAVA_OPTIONS"
echo "_JAVA_OPTIONS: $_JAVA_OPTIONS"

echo ""
echo "Unsetting restrictive Java options..."

# Unset the restrictive environment variables
unset _JAVA_OPTIONS
unset JAVA_OPTIONS

echo "Setting optimized Java options for your system..."

# Set optimized Java options for 128GB/12CPU system
export JAVA_OPTS="-Xms8g -Xmx32g -XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:+UnlockDiagnosticVMOptions -XX:MaxMetaspaceSize=1g -XX:CompressedClassSpaceSize=512m"

# Also set Spark-specific Java options
export SPARK_DRIVER_OPTS="-Xms8g -Xmx24g -XX:+UseG1GC"
export SPARK_EXECUTOR_OPTS="-XX:+UseG1GC -XX:MaxGCPauseMillis=200"

echo "New Java configuration set:"
echo "JAVA_OPTS: $JAVA_OPTS"
echo "SPARK_DRIVER_OPTS: $SPARK_DRIVER_OPTS"
echo "SPARK_EXECUTOR_OPTS: $SPARK_EXECUTOR_OPTS"

echo ""
echo "You can now run your translation process with:"
echo "python main.py --config config/optimized_config.yaml --workload-type heavy"
echo ""
echo "To make these changes permanent, add the export commands to your ~/.bashrc file"