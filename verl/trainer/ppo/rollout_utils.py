import math
from typing import List, Tuple, Dict

def find_shortest_covering_subarray(arr: List[float], tau: float) -> Tuple[float, float, float]:
    """
    Find the shortest continuous ordered subarray that covers at least τ% of data points,
    and return its mean, median, and segment length.
    
    Args:
        arr: Input one-dimensional numerical array (values range from 0 to 1)
        tau: Percentage parameter (between 0 and 100)
    
    Returns:
        tuple: (mean, median, segment_length)
    """
    if not arr:
        raise ValueError("Input array cannot be empty")
    
    if not (0 <= tau <= 100):
        raise ValueError("τ parameter must be between 0 and 100")
    
    n = len(arr)
    # Calculate the minimum number of elements to cover
    min_elements = math.ceil(n * tau / 100)
    
    if min_elements == 0:
        # If τ=0, return statistics for empty subarray (here return first element)
        return arr[0], arr[0], 0.0
    
    # Sort the array
    sorted_arr = sorted(arr)
    
    # Use sliding window to find the shortest continuous subarray
    min_length = float('inf')
    best_start = 0
    best_end = min_elements - 1
    
    # Sliding window: starting with min_elements elements
    for start in range(n - min_elements + 1):
        end = start + min_elements - 1
        current_length = sorted_arr[end] - sorted_arr[start]
        
        if current_length < min_length:
            min_length = current_length
            best_start = start
            best_end = end
    
    # Extract the shortest subarray
    shortest_subarray = sorted_arr[best_start:best_end + 1]
    
    # Calculate mean
    mean = sum(shortest_subarray) / len(shortest_subarray)
    
    # Calculate median
    subarray_length = len(shortest_subarray)
    if subarray_length % 2 == 1:
        # Odd number of elements, median is the middle element
        median = shortest_subarray[subarray_length // 2]
    else:
        # Even number of elements, median is the average of two middle elements
        mid1 = shortest_subarray[subarray_length // 2 - 1]
        mid2 = shortest_subarray[subarray_length // 2]
        median = (mid1 + mid2) / 2
    
    # Calculate segment length
    segment_length = sorted_arr[best_end] - sorted_arr[best_start]
    print(f"Segment Length: {segment_length}, Covered Index Length: {best_end - best_start + 1}")
    print(f"mean: {mean}, median: {median}")
    
    return mean, median

def analyze_distribution(arr: List[float]) -> Dict:
    """
    Analyze whether the given unordered array follows a normal distribution.
    If not normal, analyze it as a skewed distribution and calculate skewness parameters.
    
    Args:
        arr: Input one-dimensional numerical array
    
    Returns:
        dict: Dictionary containing distribution analysis results
    """
    if not arr:
        raise ValueError("Input array cannot be empty")
    
    if len(arr) < 3:
        raise ValueError("Array must contain at least 3 elements for distribution analysis")
    
    n = len(arr)
    
    # Basic statistics
    mean = sum(arr) / n
    variance = sum((x - mean) ** 2 for x in arr) / (n - 1)  # Sample variance
    std_dev = math.sqrt(variance)
    
    # Calculate skewness (third moment)
    if std_dev == 0:
        skewness = 0
    else:
        skewness = sum((x - mean) ** 3 for x in arr) / (n * std_dev ** 3)
    
    # Calculate kurtosis (fourth moment)
    if std_dev == 0:
        kurtosis = 0
    else:
        kurtosis = sum((x - mean) ** 4 for x in arr) / (n * std_dev ** 4) - 3  # Excess kurtosis
    
    # Sorted array for percentile calculations
    sorted_arr = sorted(arr)
    
    # Calculate median
    if n % 2 == 1:
        median = sorted_arr[n // 2]
    else:
        median = (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2
    
    # Calculate quartiles
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    q1 = sorted_arr[q1_idx] if q1_idx < n else sorted_arr[-1]
    q3 = sorted_arr[q3_idx] if q3_idx < n else sorted_arr[-1]
    
    # Interquartile range
    iqr = q3 - q1
    
    # Normality tests
    # 1. Skewness test (rule of thumb: |skewness| < 0.5 suggests normal)
    skewness_normal = abs(skewness) < 0.5
    
    # 2. Kurtosis test (rule of thumb: |kurtosis| < 0.5 suggests normal)
    kurtosis_normal = abs(kurtosis) < 0.5
    
    # 3. Mean vs Median test (for normal distribution, mean ≈ median)
    mean_median_diff = abs(mean - median)
    mean_median_normal = mean_median_diff < (0.1 * std_dev) if std_dev > 0 else True
    
    # 4. Empirical rule test (68-95-99.7 rule)
    # Count values within 1, 2, 3 standard deviations
    within_1_std = sum(1 for x in arr if abs(x - mean) <= std_dev)
    within_2_std = sum(1 for x in arr if abs(x - mean) <= 2 * std_dev)
    within_3_std = sum(1 for x in arr if abs(x - mean) <= 3 * std_dev)
    
    pct_1_std = within_1_std / n * 100
    pct_2_std = within_2_std / n * 100
    pct_3_std = within_3_std / n * 100
    
    # Expected percentages for normal distribution: 68%, 95%, 99.7%
    empirical_rule_test = (
        abs(pct_1_std - 68.27) < 15 and  # Allow 15% tolerance for small samples
        abs(pct_2_std - 95.45) < 10 and
        abs(pct_3_std - 99.73) < 5
    )
    
    # 5. Simple rank correlation test (replaces complex Shapiro-Wilk)
    if n >= 8:
        # Simple correlation between ranks and values
        rank_mean = (n - 1) / 2
        value_mean = mean
        
        numerator = sum((i - rank_mean) * (sorted_arr[i] - value_mean) for i in range(n))
        rank_var = sum((i - rank_mean) ** 2 for i in range(n))
        value_var = sum((x - value_mean) ** 2 for x in sorted_arr)
        
        if rank_var > 0 and value_var > 0:
            correlation = numerator / math.sqrt(rank_var * value_var)
            rank_correlation_test = abs(correlation) > 0.8
        else:
            rank_correlation_test = False
    else:
        rank_correlation_test = None
    
    # Overall normality assessment
    normality_tests = [skewness_normal, kurtosis_normal, mean_median_normal, empirical_rule_test]
    if rank_correlation_test is not None:
        normality_tests.append(rank_correlation_test)
    
    # Consider normal if majority of tests pass
    is_normal = sum(normality_tests) >= len(normality_tests) / 2
    
    # Distribution classification
    if is_normal:
        distribution_type = "Normal Distribution"
        distribution_params = {
            "mean": mean,
            "standard_deviation": std_dev,
            "variance": variance
        }
    else:
        # Classify skewed distribution
        if abs(skewness) < 0.5:
            skew_type = "Approximately Symmetric"
        elif skewness > 0.5:
            skew_type = "Right-skewed (Positive Skew)"
        else:
            skew_type = "Left-skewed (Negative Skew)"
        
        distribution_type = f"Skewed Distribution ({skew_type})"
        distribution_params = {
            "mean": mean,
            "median": median,
            "standard_deviation": std_dev,
            "variance": variance,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "q1": q1,
            "q3": q3,
            "iqr": iqr
        }
    
    # Additional distribution characteristics
    range_val = max(arr) - min(arr)
    coefficient_of_variation = (std_dev / mean * 100) if mean != 0 else float('inf')
    
    return {
        "distribution_type": distribution_type,
        "is_normal": is_normal,
        "distribution_parameters": distribution_params,
        "basic_statistics": {
            "count": n,
            "mean": mean,
            "median": median,
            "std_deviation": std_dev,
            "variance": variance,
            "minimum": min(arr),
            "maximum": max(arr),
            "range": range_val,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "coefficient_of_variation": coefficient_of_variation
        },
        "normality_tests": {
            "skewness_test": {
                "skewness": skewness,
                "is_normal": skewness_normal,
                "interpretation": "Normal" if skewness_normal else "Non-normal"
            },
            "kurtosis_test": {
                "kurtosis": kurtosis,
                "is_normal": kurtosis_normal,
                "interpretation": "Normal" if kurtosis_normal else "Non-normal"
            },
            "mean_median_test": {
                "mean_median_difference": mean_median_diff,
                "is_normal": mean_median_normal,
                "interpretation": "Normal" if mean_median_normal else "Non-normal"
            },
            "empirical_rule_test": {
                "within_1_std_pct": pct_1_std,
                "within_2_std_pct": pct_2_std,
                "within_3_std_pct": pct_3_std,
                "is_normal": empirical_rule_test,
                "interpretation": "Normal" if empirical_rule_test else "Non-normal"
            },
            "rank_correlation_test": {
                "is_normal": rank_correlation_test,
                "interpretation": "Normal" if rank_correlation_test else "Non-normal" if rank_correlation_test is not None else "Not applicable (sample too small)"
            }
        },
        "distribution_shape": {
            "skewness": skewness,
            "skewness_interpretation": (
                "Symmetric" if abs(skewness) < 0.5 else
                "Moderately skewed" if abs(skewness) < 1 else
                "Highly skewed"
            ),
            "kurtosis": kurtosis,
            "kurtosis_interpretation": (
                "Normal tail thickness" if abs(kurtosis) < 0.5 else
                "Heavy tails" if kurtosis > 0.5 else
                "Light tails"
            )
        }
    }


def print_distribution_analysis(analysis: Dict):
    """
    Print distribution analysis results in a formatted way
    
    Args:
        analysis: Result dictionary from analyze_distribution function
    """
    
    # Distribution type
    print(f"\n📊 Distribution Type: {analysis['distribution_type']}")
    print(f"🔍 Is Normal Distribution: {'Yes' if analysis['is_normal'] else 'No'}")
    
    # Distribution parameters
    print(f"\n🎯 Distribution Parameters:")
    params = analysis['distribution_parameters']
    for key, value in params.items():
        if isinstance(value, float):
            print(f"   • {key.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    # Distribution shape
    shape = analysis['distribution_shape']
    print(f"\n📐 Distribution Shape:")
    print(f"   • Skewness: {shape['skewness']:.4f} ({shape['skewness_interpretation']})")
    print(f"   • Kurtosis: {shape['kurtosis']:.4f} ({shape['kurtosis_interpretation']})")
    
    print("="*70)