import csv
import os

def process_large_csv(input_filepath, output_filepath):
    """
    Reads a large CSV file, deletes the first 22 columns,
    and adds a new header to the resulting 7-column file.

    Args:
        input_filepath (str): Path to the large input CSV file.
        output_filepath (str): Path to save the processed output CSV file.
    """
    # The header for the new 7-column file
    new_header = ["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]

    # Check if input file exists
    if not os.path.exists(input_filepath):
        print(f"Error: Input file '{input_filepath}' not found.")
        return

    print(f"Starting processing of '{input_filepath}'...")
    print(f"Output will be saved to '{output_filepath}'")

    rows_processed = 0
    try:
        # Open the input file for reading and output file for writing
        # newline='' is important for csv module to handle line endings correctly
        with open(input_filepath, 'r', newline='') as infile, \
             open(output_filepath, 'w', newline='') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # 1. Write the new header to the output file
            writer.writerow(new_header)

            # 2. Process each row from the input file
            for row in reader:
                # Original file has 29 columns (0-indexed: 0 to 28)
                # We want to keep columns from index 22 onwards
                # Column 23 is at index 22
                # Column 24 is at index 23
                # ...
                # Column 29 is at index 28
                # This slice row[22:] will give columns 23 through 29.
                
                if len(row) >= 29: # Ensure the row has enough columns
                    # Select columns from the 23rd (index 22) to the end
                    # This should result in 7 columns (29 - 22 = 7)
                    processed_row = row[22:] 
                    writer.writerow(processed_row)
                else:
                    # Handle rows that don't have the expected number of columns
                    # You might want to log this or skip them
                    print(f"Warning: Row {rows_processed + 1} has {len(row)} columns, expected at least 29. Skipping.")
                    # If you want to be strict and fail:
                    # raise ValueError(f"Row {rows_processed + 1} has insufficient columns: {len(row)}")

                rows_processed += 1
                if rows_processed % 100000 == 0:  # Print progress every 100,000 rows
                    print(f"Processed {rows_processed} rows...")

        print(f"\nProcessing complete.")
        print(f"Total rows (excluding header) processed: {rows_processed}")
        print(f"Output saved to '{output_filepath}'")

    except FileNotFoundError: # Should be caught by os.path.exists, but good practice
        print(f"Error: Input file '{input_filepath}' not found during processing.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        print(f"Processed {rows_processed} rows before error.")

if __name__ == "__main__":
    # --- Configuration ---
    # Replace with your actual file paths
    # On Windows, paths might look like: "C:\\path\\to\\your\\large_file.csv"
    # On Linux/macOS: "/path/to/your/large_file.csv"
    input_csv_file = "originalhiggs.csv" 
    output_csv_file = "originalhiggs.csv"
    # --- End Configuration ---

    process_large_csv(input_csv_file, output_csv_file)
