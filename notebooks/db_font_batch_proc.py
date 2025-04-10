from concurrent.futures import ThreadPoolExecutor
import sqlite3
import time
from pathlib import Path
from fontTools.ttLib import TTFont

def initialize_job_queue(db_path):
    """Create job queue table if it doesn't exist"""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS font_processing_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            font_path TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',  -- pending, processing, completed, failed
            attempts INTEGER DEFAULT 0,
            last_attempt TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()

class FontBatchProcessor:
    def __init__(self, db_path, max_workers=4):
        self.db_path = db_path
        self.max_workers = max_workers
        initialize_job_queue(db_path)
    
    def enqueue_fonts(self, fonts_dir):
        """Add all fonts to the processing queue"""
        fonts_dir = Path(fonts_dir)
        font_files = list(fonts_dir.glob("**/*.ttf"))
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert new fonts that aren't already in queue or database
            conn.executemany(
                "INSERT OR IGNORE INTO font_processing_queue (font_path) VALUES (?)",
                [(str(font),) for font in font_files]
            )
            conn.commit()
        
        return len(font_files)
    
    def process_queue(self):
        """Process all pending fonts in the queue"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while True:
                # Get batch of pending jobs
                jobs = self._get_pending_jobs(limit=self.max_workers * 2)
                if not jobs:
                    break
                
                # Process jobs in parallel
                futures = []
                for job in jobs:
                    future = executor.submit(
                        self._process_single_font,
                        job['id'],
                        job['font_path']
                    )
                    futures.append(future)
                
                # Wait for current batch to complete
                for future in futures:
                    future.result()
    
    def _get_pending_jobs(self, limit):
        """Get pending jobs and mark them as processing"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Use transaction to safely claim jobs
            cursor.execute("BEGIN IMMEDIATE")
            
            # Get pending jobs
            cursor.execute("""
                SELECT id, font_path FROM font_processing_queue
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT ?
            """, (limit,))
            jobs = cursor.fetchall()
            
            # Mark them as processing
            if jobs:
                job_ids = [job['id'] for job in jobs]
                cursor.execute(f"""
                    UPDATE font_processing_queue
                    SET status = 'processing',
                        attempts = attempts + 1,
                        last_attempt = CURRENT_TIMESTAMP
                    WHERE id IN ({','.join(['?']*len(job_ids))})
                """, job_ids)
                conn.commit()
            
            return jobs
    
    def _process_single_font(self, job_id, font_path):
        """Process a single font with retry logic"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Use your existing add_font function with retry logic
                from db_helpers import add_font
                success = add_font(font_path, conn)
                
                # Update job status
                if success:
                    conn.execute("""
                        UPDATE font_processing_queue
                        SET status = 'completed'
                        WHERE id = ?
                    """, (job_id,))
                    conn.commit()
                else:
                    self._mark_job_failed(job_id)
        except Exception as e:
            print(f"Error processing {font_path}: {str(e)}")
            self._mark_job_failed(job_id)
    
    def _mark_job_failed(self, job_id):
        """Mark a job as failed after max attempts"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE font_processing_queue
                SET status = CASE
                    WHEN attempts >= 3 THEN 'failed'
                    ELSE 'pending'
                END
                WHERE id = ?
            """, (job_id,))
            conn.commit()