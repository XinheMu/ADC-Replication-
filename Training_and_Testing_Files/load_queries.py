import sys
import os
from dotenv import load_dotenv
import pickle
import pandas as pd # Still useful for intermediate data handling if needed, though final output is custom
from collections import OrderedDict
import csv # For writing the custom CSV format

def convert(ALL_ATTRIBUTE_NAMES_ORDERED=0,ATTRIBUTE_METADATA=0,workload_file_path=0,output_custom_csv_path=0,shortcut=0,querytype=0):
    project_root = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded .env from {dotenv_path}")
    else:
        print(f"Warning: .env file not found at {dotenv_path}")

    data_root_env = os.getenv('DATA_ROOT')
    print(f"DATA_ROOT from environment: {data_root_env}")
    if not data_root_env:
        print("CRITICAL ERROR: DATA_ROOT not found in environment. Exiting.")
        exit(1)
    if shortcut=='advantage':
        ALL_ATTRIBUTE_NAMES_ORDERED = ['A', 'B', 'C', 'D', 'E']
        ATTRIBUTE_METADATA = {
            'A': {'minval': 0, 'maxval': 1999, 'dtype': 'int'},
            'B': {'minval': 0, 'maxval': 1999, 'dtype': 'int'},
            'C': {'minval': 0, 'maxval': 1999, 'dtype': 'int'},
            'D': {'minval': 0, 'maxval': 1999, 'dtype': 'int'},
            'E': {'minval': 0, 'maxval': 1999, 'dtype': 'int'},
        }
        print("Using defined metadata for 'advantage' dataset.")
        workload_file_path = "advantage/workload/base.pkl"
        output_custom_csv_path = "advantage_"+querytype+"set.csv"
    if shortcut=='power':
        ALL_ATTRIBUTE_NAMES_ORDERED = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
        ATTRIBUTE_METADATA = {
            'Global_active_power': {'minval': -1.0, 'maxval': 11.122, 'dtype': 'float64'},
            'Global_reactive_power': {'minval': -1.0, 'maxval': 1.39, 'dtype': 'float64'},
            'Voltage': {'minval': -1.0, 'maxval': 254.15, 'dtype': 'float64'},
            'Global_intensity': {'minval': -1.0, 'maxval': 48.4, 'dtype': 'float64'},
            'Sub_metering_1': {'minval': -1.0, 'maxval': 88, 'dtype': 'float64'},
            'Sub_metering_2': {'minval': -1.0, 'maxval': 80, 'dtype': 'float64'},
            'Sub_metering_3': {'minval': -1.0, 'maxval': 31, 'dtype': 'float64'},            
        }        
        print("Using defined metadata for 'power' dataset.")
        workload_file_path = "power/workload/base.pkl"
        output_custom_csv_path = "power_"+querytype+"set.csv"
    if shortcut=='forest':
        ALL_ATTRIBUTE_NAMES_ORDERED = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
        ATTRIBUTE_METADATA = {
            'Elevation': {'minval': 1859, 'maxval': 3858, 'dtype': 'int64'},
            'Aspect': {'minval': 0, 'maxval': 360, 'dtype': 'int64'},
            'Slope': {'minval': 0, 'maxval': 66, 'dtype': 'int64'},
            'Horizontal_Distance_To_Hydrology': {'minval': 0, 'maxval': 1397, 'dtype': 'int64'},
            'Vertical_Distance_To_Hydrology': {'minval': -173, 'maxval': 601, 'dtype': 'int64'},
            'Horizontal_Distance_To_Roadways': {'minval': 0, 'maxval': 7117, 'dtype': 'int64'},
            'Hillshade_9am': {'minval': 0, 'maxval': 254, 'dtype': 'int64'},        
            'Hillshade_Noon': {'minval': 0, 'maxval': 254, 'dtype': 'int64'},
            'Hillshade_3pm': {'minval': 0, 'maxval': 254, 'dtype': 'int64'},
            'Horizontal_Distance_To_Fire_Points': {'minval': 0, 'maxval': 7173, 'dtype': 'int64'},    
        }        
        print("Using defined metadata for 'forest' dataset.")
        workload_file_path = "forest/workload/base.pkl"
        output_custom_csv_path = "forest_"+querytype+"set.csv"
    if shortcut=='higgs':
        ALL_ATTRIBUTE_NAMES_ORDERED = ['m_jj','m_jjj','m_lv','m_jlv','m_bb','m_wbb','m_wwbb']
        ATTRIBUTE_METADATA = {
            'm_jj': {'minval': 0.074, 'maxval': 40.193, 'dtype': 'float64'},
            'm_jjj': {'minval': 0.198, 'maxval': 20.374, 'dtype': 'float64'},
            'm_lv': {'minval': 0.082, 'maxval': 7.994, 'dtype': 'float64'},
            'm_jlv': {'minval': 0.131, 'maxval': 14.263, 'dtype': 'float64'},
            'm_bb': {'minval': 0.047, 'maxval': 17.764, 'dtype': 'float64'},
            'm_wbb': {'minval': 0.294, 'maxval': 11.498, 'dtype': 'float64'},
            'm_wwbb': {'minval': 0.330, 'maxval': 8.375, 'dtype': 'float64'},
        }
        print("Using defined metadata for 'higgs' dataset.")
        workload_file_path = "higgs/workload/base.pkl"
        output_custom_csv_path = "higgs_"+querytype+"set.csv"

    # This list will store pairs of rows (low_bounds_list, high_bounds_list) for each query
    output_rows_for_csv = []

    try:
        print(f"Attempting to load workload from: {workload_file_path}")
        with open(workload_file_path, "rb") as f:
            loaded_workload_dict = pickle.load(f)
        print("Workload file loaded successfully.")

        all_queries_to_process = []
        for key in [querytype]:
            if key in loaded_workload_dict and isinstance(loaded_workload_dict[key], list):
                print(f"Adding {len(loaded_workload_dict[key])} queries from split: {key}")
                all_queries_to_process.extend(loaded_workload_dict[key])
            else:
                print(f"Split '{key}' not found or not a list in workload dictionary.")
        
        if not all_queries_to_process:
            print("No queries found to process. Exiting.")
            exit()

        print(f"Total queries to process: {len(all_queries_to_process)}")

        for i, query_obj in enumerate(all_queries_to_process):
            if not hasattr(query_obj, 'predicates'):
                print(f"Warning: Item at index {i} is not a valid Query object (no 'predicates'). Skipping.")
                continue

            query_predicates = query_obj.predicates
            
            current_query_low_bounds = []
            current_query_high_bounds = []

            for attr_name in ALL_ATTRIBUTE_NAMES_ORDERED:
                meta = ATTRIBUTE_METADATA.get(attr_name)
                attr_min_val = meta['minval']
                attr_max_val = meta['maxval']

                # Default to full range
                low_bound_for_attr = attr_min_val
                high_bound_for_attr = attr_max_val

                predicate_on_attr = query_predicates.get(attr_name)

                if predicate_on_attr is not None:
                    if isinstance(predicate_on_attr, tuple) and len(predicate_on_attr) == 2:
                        op, val = predicate_on_attr
                        
                        if pd.isna(val) and op == '=':
                            low_bound_for_attr = val 
                            high_bound_for_attr = val 
                        elif op == '=':
                            low_bound_for_attr = val
                            high_bound_for_attr = val
                        elif op == '<=':
                            high_bound_for_attr = val
                        elif op == '>=':
                            low_bound_for_attr = val
                        elif op == '<':
                            high_bound_for_attr = val 
                        elif op == '>':
                            low_bound_for_attr = val 
                        elif op == '[]':
                            if isinstance(val, tuple) and len(val) == 2:
                                low_bound_for_attr, high_bound_for_attr = val
                            else:
                                print(f"Warning: Malformed range value for {attr_name} in query {i}: {val}. Using full range.")
                        else:
                            print(f"Warning: Unknown operator '{op}' for {attr_name} in query {i}. Using full range.")
                    elif predicate_on_attr is not None:
                        print(f"Warning: Malformed predicate format for {attr_name} in query {i}: {predicate_on_attr}. Using full range.")
                
                current_query_low_bounds.append(low_bound_for_attr)
                current_query_high_bounds.append(high_bound_for_attr)
            
            output_rows_for_csv.append(current_query_low_bounds)
            output_rows_for_csv.append(current_query_high_bounds)

        if not output_rows_for_csv:
            print("No data was processed to write to CSV.")
        else:
            # Write to CSV using the csv module for no headers and custom structure
            with open(output_custom_csv_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                for row in output_rows_for_csv:
                    writer.writerow(row)
            
            print(f"Successfully converted workload to custom format: {output_custom_csv_path}")
            print(f"Generated CSV has {len(output_rows_for_csv)} rows.")
            # For verification, print the first few pairs of rows
            print("First few query representations (2 rows per query):")
            for k in range(min(4, len(output_rows_for_csv))): # Print first 4 lines (2 queries)
                print(' '.join(map(str, output_rows_for_csv[k])))


    except FileNotFoundError:
        print(f"ERROR: File not found at {workload_file_path}")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        import traceback
        traceback.print_exc()
convert(shortcut=sys.argv[1],querytype=sys.argv[2])

