
from db_font_batch_proc import FontBatchProcessor
from config.settings import DATABASE, DEBUG, DIR
from pathlib import Path
from db_helpers import init_db
from db_google_fonts_download import main
import os

MAX_WORKERS = os.cpu_count()

if __name__ == "__main__":
    print(f"Downloading fonts...")
    main()
    init_db(db_path=DATABASE['path'])

    # Initialize processor
    processor = FontBatchProcessor(
        db_path=DATABASE['path'],
        max_workers=MAX_WORKERS
    )

    # Enqueue all fonts in directory
    fonts_dir = Path(DIR['fonts'])
    total_fonts = processor.enqueue_fonts(fonts_dir)
    print(f"Enqueued {total_fonts} fonts for processing")

    # Process the queue
    print(f"Processing fonts with {MAX_WORKERS} CPU Cores...")
    processor.process_queue()