#!/usr/bin/env python3
"""
Debug script to check city configurations and IDs
"""

import os
import yaml
from city_config import CityConfigManager

def debug_city_configs():
    """Debug city configuration issues"""
    print("🔍 DEBUGGING CITY CONFIGURATIONS")
    print("=" * 50)
    
    # Check if config file exists
    config_file = "usa_city_configs.yaml"
    if not os.path.exists(config_file):
        print(f"❌ Configuration file '{config_file}' not found!")
        print("Run generate_usa_cities.py first to create the configuration file.")
        return
    
    print(f"✅ Configuration file found: {config_file}")
    
    try:
        # Load the config manager
        manager = CityConfigManager()
        print(f"📊 Loaded {len(manager.configs)} city configurations")
        
        # Check for Grand Forks specifically
        grand_forks_variants = [
            'grand_forks_nd',
            'grandforks_nd', 
            'grand forks_nd',
            'Grand Forks_ND'
        ]
        
        print(f"\n🔍 Searching for Grand Forks variants:")
        for variant in grand_forks_variants:
            if variant in manager.configs:
                print(f"   ✅ Found: {variant}")
            else:
                print(f"   ❌ Not found: {variant}")
        
        # Show all North Dakota cities
        print(f"\n🏛️ All North Dakota cities in config:")
        nd_cities = [city_id for city_id, config in manager.configs.items() 
                    if config.market_data.state_code == 'ND']
        
        if nd_cities:
            for city_id in nd_cities:
                config = manager.configs[city_id]
                print(f"   {city_id} -> {config.display_name}")
        else:
            print("   No North Dakota cities found!")
        
        # Show first 10 city IDs for reference
        print(f"\n📋 First 10 city IDs in config:")
        city_ids = list(manager.configs.keys())[:10]
        for city_id in city_ids:
            config = manager.configs[city_id]
            print(f"   {city_id} -> {config.display_name}")
        
        # Search functionality test
        print(f"\n🔎 Search test for 'grand':")
        search_results = manager.search_cities('grand')
        if search_results:
            for config in search_results:
                print(f"   Found: {config.city_id} -> {config.display_name}")
        else:
            print("   No results found")
        
        # Check the raw YAML to see the actual structure
        print(f"\n📄 Raw YAML structure check:")
        with open(config_file, 'r') as f:
            yaml_content = yaml.safe_load(f)
        
        if yaml_content and 'cities' in yaml_content:
            yaml_cities = yaml_content['cities']
            print(f"   Found {len(yaml_cities)} cities in YAML")
            
            # Look for Grand Forks in raw YAML
            grand_forks_in_yaml = [city_id for city_id in yaml_cities.keys() 
                                  if 'grand' in city_id.lower() and 'forks' in city_id.lower()]
            if grand_forks_in_yaml:
                print(f"   Grand Forks found in YAML: {grand_forks_in_yaml}")
            else:
                print("   Grand Forks not found in YAML")
                
                # Show all ND cities in YAML
                nd_in_yaml = [city_id for city_id, city_data in yaml_cities.items() 
                             if city_data.get('market_data', {}).get('state_code') == 'ND']
                print(f"   ND cities in YAML: {nd_in_yaml}")
        
    except Exception as e:
        print(f"❌ Error loading configurations: {e}")
        import traceback
        traceback.print_exc()

def test_city_id_generation():
    """Test how city IDs are generated"""
    print(f"\n🧪 TESTING CITY ID GENERATION")
    print("=" * 30)
    
    test_cities = [
        {'city': 'Grand Forks', 'state': 'ND'},
        {'city': 'New York', 'state': 'NY'},
        {'city': 'Los Angeles', 'state': 'CA'},
        {'city': 'St. Louis', 'state': 'MO'},
        {'city': 'Las Vegas', 'state': 'NV'}
    ]
    
    for city_info in test_cities:
        # Simulate the city ID generation logic
        city_name = city_info['city']
        state_code = city_info['state']
        
        # Standard transformation (like in city_config.py)
        city_id = f"{city_name.lower().replace(' ', '_').replace('.', '').replace('-', '_')}_{state_code.lower()}"
        print(f"   {city_info['city']}, {city_info['state']} -> {city_id}")

def fix_suggestions():
    """Provide fix suggestions"""
    print(f"\n💡 FIX SUGGESTIONS")
    print("=" * 20)
    print("1. Check if generate_usa_cities.py was run successfully")
    print("2. Verify the city ID format matches what's expected")
    print("3. Check if Grand Forks was included in the cities_data list")
    print("4. Run this command to regenerate configs:")
    print("   python generate_usa_cities.py")
    print("5. Check the dynamic_data_loader.py to see what city_id it's looking for")

if __name__ == "__main__":
    debug_city_configs()
    test_city_id_generation()
    fix_suggestions()