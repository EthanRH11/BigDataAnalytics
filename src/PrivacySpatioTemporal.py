import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
import os
from scipy.spatial.distance import cdist
from scipy import stats
import math

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("Successfully imported matplotlib with AGG backend")
except ImportError:
    print("Warning: matplotlib could not be imported. Visualization will be skipped")
    ply = None

#1. Load Data and basic queries
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Loaded {len(df)} data points for {df['user_id'].nunique()} users")
    return df

def basic_queries(df):
    print("\n--- BASIC QUERIES (NO PRIVACY PROTECTION) ---")
    # Count points by location
    location_counts = df.groupby(['latitude', 'longitude']).size().reset_index(name='count')
    print(f"Most popular location: {location_counts.loc[location_counts['count'].idxmax()]}")
    
    # Count points by user
    user_counts = df.groupby('user_id').size().reset_index(name='count')
    print(f"User with most points: {user_counts.loc[user_counts['count'].idxmax()]}")
    
    # Find a populated area by checking the min/max range of data
    lat_min = df['latitude'].min()
    lat_max = df['latitude'].max()
    lon_min = df['longitude'].min()
    lon_max = df['longitude'].max()
    
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    lat_center = (lat_max + lat_min) / 2
    lon_center = (lon_max + lon_min) / 2
    
    sample_lat_min = lat_center - lat_range * 0.05
    sample_lat_max = lat_center + lat_range * 0.05
    sample_lon_min = lon_center - lon_range * 0.05
    sample_lon_max = lon_center + lon_range * 0.05
    
    # Query points in the selected area
    area_points = df[(df['latitude'] >= sample_lat_min) & (df['latitude'] <= sample_lat_max) &
                     (df['longitude'] >= sample_lon_min) & (df['longitude'] <= sample_lon_max)]
    
    print(f"Data spans latitude: {lat_min:.6f} to {lat_max:.6f}, longitude: {lon_min:.6f} to {lon_max:.6f}")
    print(f"Querying central region: lat {sample_lat_min:.6f}-{sample_lat_max:.6f}, lon {sample_lon_min:.6f}-{sample_lon_max:.6f}")
    print(f"Points in queried area: {len(area_points)}")
    
    # Query points in a specific time range
    start_time = df['timestamp'].min() + timedelta(days=1)
    end_time = start_time + timedelta(hours=12)
    time_points = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    print(f"Points in time range {start_time} to {end_time}: {len(time_points)}")
    
    return location_counts, user_counts, area_points, time_points


#2. Location obfuscation with differential privacy

def add_laplace_noise(df, epsilon=1.0, column_scales=None):
    if column_scales is None:
        column_scales = {
            'latitude': 0.001,
            'longitude': 0.001
        }

        df_noisy = df.copy()

        for column, sensitivity in column_scales.items():
            scale = sensitivity / epsilon
            noise = np.random.laplace(0, scale, size=len(df))
            df_noisy[column] = df_noisy[column] + noise

        return df_noisy
    
def query_with_laplace_noise(df, epsilons=[0.1, 0.5, 1.0, 2.0]):
    print(f"\n--- QUERIES WITH LAPLACE NOISE (DIFFERENTIAL PRIVACY) ---")

     # Get data bounds to query a populated area
    lat_min = df['latitude'].min()
    lat_max = df['latitude'].max()
    lon_min = df['longitude'].min()
    lon_max = df['longitude'].max()
    
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    lat_center = (lat_max + lat_min) / 2
    lon_center = (lon_max + lon_min) / 2
    
    sample_lat_min = lat_center - lat_range * 0.05
    sample_lat_max = lat_center + lat_range * 0.05
    sample_lon_min = lon_center - lon_range * 0.05
    sample_lon_max = lon_center + lon_range * 0.05
    
    # Get original results for comparison
    area_points_original = df[(df['latitude'] >= sample_lat_min) & 
                             (df['latitude'] <= sample_lat_max) &
                             (df['longitude'] >= sample_lon_min) & 
                             (df['longitude'] <= sample_lon_max)]
    
    print(f"Querying central region: lat {sample_lat_min:.6f}-{sample_lat_max:.6f}, lon {sample_lon_min:.6f}-{sample_lon_max:.6f}")
    print(f"Original points in area: {len(area_points_original)}")
    
    # Test different epsilon values
    print("\nTesting multiple privacy levels (epsilon values):")
    print(f"{'Epsilon':<10} {'Points in Area':<15} {'Difference':<15} {'Privacy Level':<15}")
    print("-" * 60)
    
    results = {}
    for epsilon in epsilons:
        # Apply Laplace noise with this epsilon
        df_noisy = add_laplace_noise(df, epsilon)
        
        # Execute the query on noisy data
        area_points_noisy = df_noisy[(df_noisy['latitude'] >= sample_lat_min) & 
                                    (df_noisy['latitude'] <= sample_lat_max) &
                                    (df_noisy['longitude'] >= sample_lon_min) & 
                                    (df_noisy['longitude'] <= sample_lon_max)]
        
        # Calculate difference
        difference = len(area_points_noisy) - len(area_points_original)
        percent_diff = (difference / len(area_points_original) * 100) if len(area_points_original) > 0 else float('inf')
        
        if epsilon <= 0.1:
            privacy = "Very High"
        elif epsilon <= 0.5:
            privacy = "High"
        elif epsilon <= 1.0:
            privacy = "Medium"
        else:
            privacy = "Low"
            
        print(f"{epsilon:<10} {len(area_points_noisy):<15} {difference:<+15} {privacy:<15}")
        
        # Store result for return
        results[epsilon] = df_noisy
    
    print("\nNote: Lower epsilon = more privacy but less accuracy")
    
    return results.get(1.0, results[list(results.keys())[0]])

#3 Spatial K-Anonymity

def apply_spatial_kanonymity(df, k=5, grid_size=0.01):
    # Create a copy
    df_anonymized = df.copy()
    
    df_anonymized['lat_grid'] = (df_anonymized['latitude'] // grid_size) * grid_size
    df_anonymized['lon_grid'] = (df_anonymized['longitude'] // grid_size) * grid_size
    
    grid_counts = df_anonymized.groupby(['lat_grid', 'lon_grid']).size()
    
    valid_cells = grid_counts[grid_counts >= k].reset_index()
    
    df_anonymized = pd.merge(
        df_anonymized, 
        valid_cells, 
        on=['lat_grid', 'lon_grid'],
        how='inner'
    )
    
    df_anonymized['latitude'] = df_anonymized['lat_grid']
    df_anonymized['longitude'] = df_anonymized['lon_grid']
    
    df_anonymized = df_anonymized.drop(['lat_grid', 'lon_grid'], axis=1)
    
    return df_anonymized

def query_with_kanonymity(df, k=5):
    print("\n--- QUERIES WITH SPATIAL K-ANONYMITY ---")
    
    # Apply k-anonymity
    df_kanon = apply_spatial_kanonymity(df, k)
    
    print(f"Original data points: {len(df)}")
    print(f"Points after k-anonymity (k={k}): {len(df_kanon)}")
    print(f"Points suppressed: {len(df) - len(df_kanon)}")
    
    orig_locations = df[['latitude', 'longitude']].drop_duplicates().shape[0]
    kanon_locations = df_kanon[['latitude', 'longitude']].drop_duplicates().shape[0]
    print(f"Original distinct locations: {orig_locations}")
    print(f"Distinct locations after k-anonymity: {kanon_locations}")
    
    return df_kanon

# 4. temporal cloaking

def apply_temporal_cloaking(df, time_interval=3600):
    # Create a copy
    df_cloaked = df.copy()
    
    # Round timestamps to the specified interval
    df_cloaked['timestamp'] = df_cloaked['timestamp'].dt.floor(f"{time_interval}s")
    
    return df_cloaked

def query_with_temporal_cloaking(df, interval_hours=1):
    """Demonstrate queries with temporal cloaking"""
    print("\n--- QUERIES WITH TEMPORAL CLOAKING ---")
    
    interval_seconds = interval_hours * 3600
    df_cloaked = apply_temporal_cloaking(df, interval_seconds)
    
    # Compare unique timestamps before and after cloaking
    orig_timestamps = df['timestamp'].nunique()
    cloaked_timestamps = df_cloaked['timestamp'].nunique()
    
    print(f"Original unique timestamps: {orig_timestamps}")
    print(f"Unique timestamps after {interval_hours}-hour cloaking: {cloaked_timestamps}")
    print(f"Temporal resolution reduced by factor of: {orig_timestamps / cloaked_timestamps:.1f}x")
    
    return df_cloaked

# 5. Query Result Perturbation

def perturb_query_result(result, epsilon=1.0):
    sensitivity = 1
    scale = sensitivity / epsilon
    
    # Add Laplace noise
    noise = np.random.laplace(0, scale)
    
    perturbed_result = max(0, round(result + noise))
    
    return perturbed_result

def demonstrate_result_perturbation(df):
    """Demonstrate perturbing query results instead of data"""
    print("\n--- QUERY RESULT PERTURBATION ---")
    
    queries = [
        ("Users who visited parks", len(df[df['poi_type'] == 'park']['user_id'].unique())),
        ("Points during morning hours (6-10 AM)", len(df[df['timestamp'].dt.hour.between(6, 10)])),
        ("Users with more than 10 points", len(df.groupby('user_id').filter(lambda x: len(x) > 10)['user_id'].unique())),
    ]
    
    # Test different privacy levels
    epsilons = [0.1, 0.5, 1.0, 2.0]
    
    print(f"{'Query':<40} {'Actual':<10} " + " ".join([f"ε={e:<8}" for e in epsilons]))
    print("-" * 80)
    
    for query_name, actual_result in queries:
        perturbed_results = [perturb_query_result(actual_result, epsilon) for epsilon in epsilons]
        print(f"{query_name:<40} {actual_result:<10} " + " ".join([f"{result:<10}" for result in perturbed_results]))

# 6. Visualization functions

def visualize_privacy_effects(df, df_noisy, df_kanon):
    """Visualize the effects of privacy techniques on the data"""
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.scatter(df['longitude'], df['latitude'], s=1, alpha=0.5)
    plt.title('Original Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Noisy data (Differential Privacy)
    plt.subplot(1, 3, 2)
    plt.scatter(df_noisy['longitude'], df_noisy['latitude'], s=1, alpha=0.5)
    plt.title('Data with Laplace Noise')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # K-anonymized data
    plt.subplot(1, 3, 3)
    plt.scatter(df_kanon['longitude'], df_kanon['latitude'], s=1, alpha=0.5)
    plt.title('K-anonymized Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.tight_layout()
    plt.savefig('privacy_effects_visualization.png')
    print("\nVisualization saved as 'privacy_effects_visualization.png'")

def visualize_temporal_cloaking(df_original, df_cloaked, interval_hours):
    plt.figure(figsize=(12, 6))

    original_times = pd.to_datetime(df_original['timestamp']).sort_values()
    cloaked_times = pd.to_datetime(df_cloaked['timestamp']).sort_values()

    # Create two subplots
    plt.subplot(2, 1, 1)
    plt.hist(original_times, bins=50, alpha=0.7, color='blue')
    plt.title('Original Timestamp Distribution')
    plt.ylabel('Frequency')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    plt.hist(cloaked_times, bins=50, alpha=0.7, color='green')
    plt.title(f'Timestamp Distribution After {interval_hours}-Hour Cloaking')
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('temporal_cloaking_visualization.png')
    print("\nTemporal cloaking visualization saved as 'temporal_cloaking_visualization.png'")


def visualize_privacy_heatmaps(df, df_noisy, df_kanon):

    plt.figure(figsize=(15, 5))
    
    # Create a grid for the heatmap
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    
    lon_padding = (lon_max - lon_min) * 0.05
    lat_padding = (lat_max - lat_min) * 0.05
    
    lon_min -= lon_padding
    lon_max += lon_padding
    lat_min -= lat_padding
    lat_max += lat_padding
    
    lon_bins = np.linspace(lon_min, lon_max, 100)
    lat_bins = np.linspace(lat_min, lat_max, 100)
    
    # Function to create histogram2d data
    def create_heatmap_data(df_data):
        hist, _, _ = np.histogram2d(
            df_data['longitude'], df_data['latitude'], 
            bins=[lon_bins, lat_bins]
        )
        return hist.T  
    
    hist_original = create_heatmap_data(df)
    hist_noisy = create_heatmap_data(df_noisy)
    hist_kanon = create_heatmap_data(df_kanon)
    
    titles = ['Original Data Density', 'DP Noisy Data Density', 'K-anonymized Data Density']
    data_sets = [hist_original, hist_noisy, hist_kanon]
    
    for i, (title, data) in enumerate(zip(titles, data_sets)):
        ax = plt.subplot(1, 3, i+1)
        
        im = ax.imshow(data, cmap='hot', origin='lower', aspect='auto', 
                    extent=[lon_min, lon_max, lat_min, lat_max])
        
        plt.title(title)
        plt.xlabel('Longitude')
        if i == 0:
            plt.ylabel('Latitude')
        
        # Add colorbar
        plt.colorbar(im)
    
    plt.tight_layout()
    plt.savefig('privacy_heatmaps.png')
    print("\nHeatmap visualization saved as 'privacy_heatmaps.png'")

def visualize_privacy_utility_tradeoff(df, epsilons=[0.1, 0.5, 1.0, 2.0, 5.0], k_values=[3, 5, 10, 15, 20], 
                                      time_intervals=[0.5, 1, 2, 6, 12]):
    """
    Create visualizations showing the privacy-utility tradeoff for different
    privacy techniques with various parameter settings.
    """
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    
    dp_utility = []
    dp_distortion = []
    
    sample_size = min(1000, len(df))
    df_sample = df.sample(sample_size)
    
    for epsilon in epsilons:
        df_noisy = add_laplace_noise(df_sample, epsilon)
        
        orig_coords = df_sample[['latitude', 'longitude']].values
        noisy_coords = df_noisy[['latitude', 'longitude']].values
        
        distances = np.sqrt(np.sum((orig_coords - noisy_coords)**2, axis=1))
        avg_distortion = np.mean(distances)
        
        dp_distortion.append(avg_distortion)
        dp_utility.append(1 / (1 + avg_distortion))  # Higher distortion = lower utility
    
    plt.plot(epsilons, dp_utility, 'o-', label='Utility')
    plt.plot(epsilons, [1/x for x in dp_distortion], 's-', label='1/Distortion')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.title('Differential Privacy Tradeoff: Utility vs. Privacy Budget (ε)')
    plt.xlabel('Privacy Budget (ε) - Higher ε means less privacy')
    plt.ylabel('Utility Metric')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    
    k_data_retention = []
    k_location_retention = []
    
    for k in k_values:
        df_kanon = apply_spatial_kanonymity(df, k)
        
        data_retention = len(df_kanon) / len(df)
        location_retention = df_kanon[['latitude', 'longitude']].drop_duplicates().shape[0] / \
                            df[['latitude', 'longitude']].drop_duplicates().shape[0]
        
        k_data_retention.append(data_retention)
        k_location_retention.append(location_retention)
    
    plt.plot(k_values, k_data_retention, 'o-', label='Data Point Retention')
    plt.plot(k_values, k_location_retention, 's-', label='Location Retention')
    plt.title('K-Anonymity Tradeoff: Data Retention vs. Privacy Level (k)')
    plt.xlabel('Privacy Level (k) - Higher k means more privacy')
    plt.ylabel('Retention Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    
    time_resolution = []
    time_granularity = []
    
    for interval in time_intervals:
        df_cloaked = apply_temporal_cloaking(df, interval * 3600)
        
        # Resolution reduction (higher is more privacy)
        resolution_reduction = df['timestamp'].nunique() / df_cloaked['timestamp'].nunique()
        time_granularity.append(interval)  
        time_resolution.append(resolution_reduction)
    
    plt.plot(time_intervals, time_resolution, 'o-', label='Temporal Resolution Reduction')
    plt.title('Temporal Cloaking Tradeoff: Resolution vs. Time Interval')
    plt.xlabel('Cloaking Interval (Hours) - Higher interval means more privacy')
    plt.ylabel('Temporal Resolution Reduction Factor')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('privacy_utility_tradeoff.png')
    print("\nPrivacy-utility tradeoff visualization saved as 'privacy_utility_tradeoff.png'")



# Main function

def main():
    np.random.seed(42)
    
    # Parameters
    data_file = "../data/data/synthetic_spatiotemporal_data.csv"
    epsilons = [0.1, 0.5, 1.0, 2.0]     
    k = 5  
    time_interval = 2
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found!")
        print("Please run the data generator script first.")
        return
    
    # Load data
    df = load_data(data_file)
    
    # 1. Basic queries without privacy
    basic_queries(df)
    
    # 2. Queries with Laplace noise (differential privacy)
    df_noisy = query_with_laplace_noise(df, epsilons)
    
    # 3. Queries with spatial k-anonymity
    df_kanon = query_with_kanonymity(df, k)
    
    # 4. Queries with temporal cloaking
    df_cloaked = query_with_temporal_cloaking(df, time_interval)
    
    # 5. Query result perturbation
    demonstrate_result_perturbation(df)
    
    # 6. Visualize privacy effects
    if plt is not None:
        visualize_privacy_effects(df, df_noisy, df_kanon)
    else:
        print("\nSkipping visualization due to matplotlib import issue")
        print("To enable visualization, please troubleshoot matplotlib installation")
    # 7. More visualization methods
    if plt is not None:
        print("\n--- Generating additional visualization ---")

        visualize_temporal_cloaking(df, df_cloaked, time_interval)

        visualize_privacy_heatmaps(df, df_noisy, df_kanon)

        visualize_privacy_utility_tradeoff(df, epsilons)
    else:
        print("\nSkipping additional visualization")

    print("\nAll privacy-preserving query demonstrations completed.")
    print("You can now analyze the results to compare the different techniques.")

if __name__ == "__main__":
    main()