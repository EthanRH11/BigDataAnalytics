import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

"""
Synthetic Spatiotemporal Data Generator
"""

def generate_synthetic_data(
        num_users = 100,
        num_points_per_user = 50,
        start_date = datetime(2025, 1, 1),
        end_date = datetime(2025, 1, 31),
        region_center = (41.15361, -81.35806), #kent coords
        region_radius = 0.2,
        output_dir = "data",
        time_pattern = "random",
        movement_pattern = "random",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_data = []

    user_ids = [f"user_{i:04d}" for i in range(1, num_users + 1)]
    #time period in hours
    time_period = int((end_date - start_date).total_seconds() / 3600)

    for user_id in user_ids:
        points_for_this_user = max(5, np.random.poisson(num_points_per_user))

        home_lat = region_center[0] + np.random.uniform(-region_radius, region_radius)
        home_lon = region_center[1] + np.random.uniform(-region_radius, region_radius)

        if movement_pattern == "home_work":
            work_distance = np.random.uniform(0.01, 0.1)  # in degrees
            work_angle = np.random.uniform(0, 2*np.pi)
            work_lat = home_lat + work_distance * np.cos(work_angle)
            work_lon = home_lon + work_distance * np.sin(work_angle)
        
        if time_pattern == "random":
            timestamps = [start_date + timedelta(hours=np.random.uniform(0, time_period))
                          for _ in range(points_for_this_user)]
            timestamps.sort() #sort chronologically

        elif time_pattern == "daily":
            days = np.random.choice(range((end_date - start_date).days + 1), points_for_this_user)
            hours = np.random.normal(14, 5, points_for_this_user)
            hours = np.clip(hours, 8, 22).astype(int)
            timestamps = [start_date + timedelta(days=day, hours=hour)
                          for day, hour in zip(days, hours)]
            timestamps.sort()
            
        elif time_pattern == "weekly":
            weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
            day_offsets = []
            for _ in range(points_for_this_user):
                week = np.random.randint(0, (end_date - start_date).days // 7 + 1)
                weekday = np.random.choice(range(7), p=[w/sum(weights) for w in weights])
                hour = np.random.normal(14, 5)
                hour = np.clip(hour, 8, 22)
                day_offsets.append(week * 7 + weekday + hour/24)
                
            timestamps = [start_date + timedelta(days=offset) for offset in sorted(day_offsets)]
        
        # Generate locations based on the specified pattern
        lats = []
        lons = []
        
        if movement_pattern == "random":
            # Completely random within the region
            lats = [region_center[0] + np.random.uniform(-region_radius, region_radius) 
                   for _ in range(points_for_this_user)]
            lons = [region_center[1] + np.random.uniform(-region_radius, region_radius)
                   for _ in range(points_for_this_user)]
                   
        elif movement_pattern == "home_work":
            # Points clustered around home and work locations
            for _ in range(points_for_this_user):
                if np.random.random() < 0.4:  # 40% chance of being at home
                    lat = home_lat + np.random.normal(0, 0.001)  # ~100m std dev
                    lon = home_lon + np.random.normal(0, 0.001)
                elif np.random.random() < 0.7:  # 42% chance of being at work (0.7 * 0.6)
                    lat = work_lat + np.random.normal(0, 0.001)
                    lon = work_lon + np.random.normal(0, 0.001)
                else:  # 18% chance of being elsewhere
                    lat = region_center[0] + np.random.uniform(-region_radius, region_radius)
                    lon = region_center[1] + np.random.uniform(-region_radius, region_radius)
                lats.append(lat)
                lons.append(lon)
                
        elif movement_pattern == "clustered":
            # Generate a few random clusters for this user
            num_clusters = np.random.randint(3, 6)
            cluster_lats = [region_center[0] + np.random.uniform(-region_radius, region_radius) 
                          for _ in range(num_clusters)]
            cluster_lons = [region_center[1] + np.random.uniform(-region_radius, region_radius)
                          for _ in range(num_clusters)]
            cluster_weights = np.random.dirichlet(np.ones(num_clusters))  # random distribution of visits
            
            for _ in range(points_for_this_user):
                cluster_idx = np.random.choice(range(num_clusters), p=cluster_weights)
                lat = cluster_lats[cluster_idx] + np.random.normal(0, 0.002)
                lon = cluster_lons[cluster_idx] + np.random.normal(0, 0.002)
                lats.append(lat)
                lons.append(lon)
        
        # Generate additional attributes
        pois = ['home', 'work', 'restaurant', 'shop', 'park', 'gym', 'cafe', 'grocery', 'other']
        poi_weights = [0.3, 0.3, 0.1, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]  # probability distribution
        activities = ['stationary', 'walking', 'driving', 'cycling']
        activity_weights = [0.6, 0.2, 0.15, 0.05]
        
        for i in range(points_for_this_user):
            point_data = {
                'user_id': user_id,
                'timestamp': timestamps[i],
                'latitude': lats[i],
                'longitude': lons[i],
                'poi_type': np.random.choice(pois, p=poi_weights),
                'activity': np.random.choice(activities, p=activity_weights),
                'duration_minutes': np.random.exponential(30),  # exponential distribution with mean 30 minutes
                'speed_kmh': np.random.exponential(5),  # exponential with mean 5 km/h
                'accuracy_meters': np.random.uniform(5, 50)  # uniform between 5-50 meters
            }
            all_data.append(point_data)
    
    # Convert to DataFrame and sort by timestamp
    df = pd.DataFrame(all_data)
    df = df.sort_values('timestamp')
    
    # Save the full dataset
    df.to_csv(f"{output_dir}/synthetic_spatiotemporal_data.csv", index=False)
    
    # Save individual user trajectories
    for user_id in user_ids:
        user_df = df[df['user_id'] == user_id]
        user_df.to_csv(f"{output_dir}/user_{user_id.split('_')[1]}_trajectory.csv", index=False)
    
    # Generate summary statistics
    summary = {
        'total_users': len(user_ids),
        'total_points': len(df),
        'avg_points_per_user': len(df) / len(user_ids),
        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
        'geographic_bounds': {
            'min_lat': df['latitude'].min(),
            'max_lat': df['latitude'].max(), 
            'min_lon': df['longitude'].min(),
            'max_lon': df['longitude'].max()
        }
    }
    
    # Save summary as text file
    with open(f"{output_dir}/dataset_summary.txt", "w") as f:
        for key, value in summary.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"Generated {len(df)} data points for {len(user_ids)} users")
    print(f"Data saved to {output_dir}/ directory")
    
    return df

if __name__ == "__main__":
    generate_synthetic_data(
        num_users=100,                         # 50 distinct users
        num_points_per_user=100,              # ~100 points per user
        start_date=datetime(2023, 1, 1),      # Data starts Jan 1, 2023
        end_date=datetime(2023, 1, 31),       # Ends Jan 31, 2023
        region_center=(40.7128, -74.0060),    # Centered around NYC
        region_radius=0.1,                    # ~11km radius
        output_dir="data",                    # Save to data/ directory
        time_pattern="weekly",                # Follow weekly patterns
        movement_pattern="home_work"          # Home-work movement pattern
    )
               