#Capstone: MSBD5003x Project: Sample Sort
import os
os.environ['PYSPARK_PYTHON'] = 'python3.9'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3.9'
from pyspark import SparkContext
from pyspark.sql import SparkSession

# Check if a SparkContext already exists
if SparkContext._active_spark_context:
    SparkContext._active_spark_context.stop()

# Set SPARK_LOCAL_IP to your machine's IP address
os.environ['SPARK_LOCAL_IP'] = '192.168.1.245'  # Replace with your actual IP if different



# Import necessary modules
from pyspark import SparkContext
import random

# Initialize SparkContext
sc = SparkContext("local[*]", "SampleSort")

# Define the parameters
n = 100  # Size of the RDD to be sorted
p = 8    # Number of partitions/workers
s = 4    # Sample size factor

# Create an RDD with random integers
rdd = sc.parallelize([random.randint(1, n*100) for _ in range(n)], p)

# Take a sample of size s*p from the RDD
sample = rdd.takeSample(False, s * p)

# Sort the sample
sample.sort()

# Determine the splitters
splitters = [sample[i * s] for i in range(1, p)]
# Optionally add a 0 at the beginning
splitters = [0] + splitters

# Broadcast the splitters if desired (optional)
# splitters_broadcast = sc.broadcast(splitters)

def split_partition(partition):
    # Convert partition to a list
    items = list(partition)
    # Initialize buckets for each splitter
    buckets = [[] for _ in range(p)]
    for item in items:
        # Perform binary search to find the correct bucket
        low = 0
        high = len(splitters) - 1
        while low <= high:
            mid = (low + high) // 2
            if splitters[mid] <= item < (splitters[mid + 1] if mid + 1 < len(splitters) else float('inf')):
                buckets[mid].append(item)
                break
            elif item < splitters[mid]:
                high = mid - 1
            else:
                low = mid + 1
    # Yield bucket number and items
    for idx, bucket in enumerate(buckets):
        yield (idx, bucket)

# Split the items in each partition
split_rdd = rdd.mapPartitions(split_partition).flatMap(lambda x: [x])

# Group items by bucket number
grouped_rdd = split_rdd.groupByKey()

# Sort items in each bucket
sorted_buckets = grouped_rdd.mapValues(lambda lists: sorted([item for sublist in lists for item in sublist]))

# Concatenate sorted buckets
sorted_rdd = sorted_buckets.sortByKey().flatMap(lambda x: x[1])

# Collect the sorted result
sorted_list = sorted_rdd.collect()

# Print the sorted list
#sorted_buckets2 =sorted_buckets.collect()
#print(sorted_buckets2)
print("Sorted List:")
print(sorted_list)

# Stop the SparkContext
sc.stop()

