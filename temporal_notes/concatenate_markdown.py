import os
from pathlib import Path
import re
from typing import List, Tuple

def extract_page_number(filename: str) -> int:
    """Extract page number from filename like demo_xxxx_page_x.md"""
    match = re.search(r'page_(\d+)\.md$', filename)
    return int(match.group(1)) if match else 0

def concatenate_markdown_files(processed_data_dir: str, output_dir: str) -> None:
    """
    Navigate through subfolders and concatenate markdown files by document.
    
    Args:
        processed_data_dir: Directory containing layout_results_* folders
        output_dir: Directory to save concatenated files
    """
    processed_path = Path(processed_data_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all layout_results folders
    layout_folders = [f for f in processed_path.iterdir() 
                     if f.is_dir() and f.name.startswith('layout_results_')]
    
    if not layout_folders:
        print(f"No layout_results folders found in {processed_data_dir}")
        return
    
    print(f"Found {len(layout_folders)} document folders to process")
    
    for folder in layout_folders:
        print(f"Processing: {folder.name}")
        
        # Find all markdown files in this folder
        md_files = list(folder.glob("demo_*.md"))
        
        if not md_files:
            print(f"  No markdown files found in {folder.name}")
            continue
        
        # Sort files by page number
        md_files.sort(key=lambda x: extract_page_number(x.name))
        
        # Extract document ID from folder name
        doc_id = folder.name.replace('layout_results_', '')
        
        # Concatenate all pages
        concatenated_content = []
        concatenated_content.append(f"# Document {doc_id}\n\n")
        
        for md_file in md_files:
            page_num = extract_page_number(md_file.name)
            
            # Add page separator
            concatenated_content.append(f"## Page {page_num}\n\n")
            
            # Read and add file content
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Remove filepath comments if present
                    content = re.sub(r'^//\s*filepath:.*\n', '', content, flags=re.MULTILINE)
                    concatenated_content.append(content.strip())
                    concatenated_content.append("\n\n")
            except Exception as e:
                print(f"  Error reading {md_file.name}: {e}")
                continue
        
        # Save concatenated file
        output_file = output_path / f"document_{doc_id}.md"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(concatenated_content))
            print(f"  Saved: {output_file}")
        except Exception as e:
            print(f"  Error saving {output_file}: {e}")

def main():
    """Main function to run the concatenation script."""
    processed_data_dir = "data/processed_data"
    output_dir = "data/concatenated_docs"
    
    concatenate_markdown_files(processed_data_dir, output_dir)
    print("Concatenation completed!")

if __name__ == "__main__":
    main()