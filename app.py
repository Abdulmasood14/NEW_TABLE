#!/usr/bin/env python3
"""
Enhanced PDF Table Extractor with Streamlit Web Interface
Processes ALL pages with AI-powered table recognition using Google Gemini 2.0 Flash
Specialized for financial statements with Quarter & Nine Months format support
"""

import os
import base64
import pandas as pd
import google.generativeai as genai
from pathlib import Path
import json
import re
from typing import List, Dict, Optional, Any
import io
import fitz  # PyMuPDF
import platform
import subprocess
import sys
import streamlit as st
from datetime import datetime
import tempfile
import zipfile
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPDFTableExtractor:
    """
    Enhanced PDF Table Extractor with ALL pages processing and financial statement specialization
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Enhanced PDF Table Extractor with Gemini 2.0 Flash
        
        Args:
            api_key (str): Your Google AI API key
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Initialize Gemini 2.0 Flash model
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Base output directory - will be set per PDF
        self.base_output_dir = Path("extracted_tables")
        self.base_output_dir.mkdir(exist_ok=True)
        self.output_dir = None  # Will be set when processing PDF
        
        # Check available PDF processing methods
        self.check_dependencies()
    
    def extract_pdf_title(self, pdf_path: str) -> str:
        """
        Extract title from PDF metadata or first page content with enhanced detection
        Optimized for financial statements and company reports
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted and sanitized title
        """
        try:
            doc = fitz.open(pdf_path)
            
            # First try to get title from metadata
            metadata = doc.metadata
            if metadata and metadata.get('title'):
                title = metadata['title'].strip()
                if title and len(title) > 3:  # Valid title
                    doc.close()
                    return self.sanitize_directory_name(title)
            
            # Enhanced title extraction from first page content
            if len(doc) > 0:
                first_page = doc[0]
                blocks = first_page.get_text("dict")
                
                title_candidates = []
                page_height = first_page.rect.height
                
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            bbox = line["bbox"]
                            y_pos = bbox[1]  # y coordinate
                            
                            # Consider text in upper 40% of page for better title detection
                            if y_pos < page_height * 0.4:
                                for span in line.get("spans", []):
                                    text = span.get("text", "").strip()
                                    font_size = span.get("size", 0)
                                    
                                    # Enhanced title detection for financial statements
                                    if (text and len(text) > 8 and 
                                        font_size >= 10 and 
                                        not text.lower().startswith(('page', 'confidential', 'draft', 'notes:', 'source:'))):
                                        
                                        # Boost score for financial statement keywords
                                        score_boost = 0
                                        financial_keywords = [
                                            'financial results', 'audited results', 'unaudited', 
                                            'consolidated', 'standalone', 'quarter', 'nine months',
                                            'year ended', 'statement', 'company limited', 'ltd',
                                            'balance sheet', 'profit loss', 'cash flow'
                                        ]
                                        
                                        text_lower = text.lower()
                                        for keyword in financial_keywords:
                                            if keyword in text_lower:
                                                score_boost += 2
                                        
                                        title_candidates.append((text, font_size + score_boost, y_pos))
                
                # Sort by enhanced score (font size + keyword boost) and position
                title_candidates.sort(key=lambda x: (-x[1], x[2]))
                
                if title_candidates:
                    potential_title = title_candidates[0][0]
                    potential_title = re.sub(r'\s+', ' ', potential_title.strip())
                    
                    # Clean up common header patterns
                    potential_title = re.sub(r'^(COMPANY|CORPORATION|LIMITED|LTD|INC)[\s:]+', '', 
                                           potential_title, flags=re.IGNORECASE)
                    
                    if len(potential_title) > 5:
                        doc.close()
                        return self.sanitize_directory_name(potential_title)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting PDF title: {e}")
            if 'st' in globals():
                st.error(f"Error extracting PDF title: {e}")
        
        # Fallback to filename
        pdf_name = Path(pdf_path).stem
        return self.sanitize_directory_name(pdf_name)
    
    def sanitize_directory_name(self, name: str) -> str:
        """
        Sanitize a string to be used as a directory name
        
        Args:
            name (str): Original name
            
        Returns:
            str: Sanitized directory name
        """
        # Remove or replace invalid characters for directory names
        sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
        sanitized = re.sub(r'\s+', ' ', sanitized)
        sanitized = sanitized.strip(' .')
        
        # Limit length to avoid filesystem issues
        if len(sanitized) > 100:
            sanitized = sanitized[:100].strip()
        
        if not sanitized:
            sanitized = "Untitled_PDF"
        
        return sanitized
    
    def setup_output_directory(self, pdf_path: str):
        """
        Setup output directory based on PDF title
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        pdf_title = self.extract_pdf_title(pdf_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{pdf_title}_{timestamp}"
        
        self.output_dir = self.base_output_dir / dir_name
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Created output directory: {self.output_dir}")
        if 'st' in globals():
            st.info(f"📁 Created output directory: {self.output_dir}")
            st.info(f"📄 PDF Title detected: {pdf_title}")
    
    def check_dependencies(self):
        """Check and report available PDF processing methods"""
        try:
            import fitz
            logger.info("PyMuPDF available for PDF processing")
            if 'st' in globals():
                st.success("✓ PyMuPDF available for PDF processing")
        except ImportError:
            error_msg = "PyMuPDF not available. Please install: pip install PyMuPDF"
            logger.error(error_msg)
            if 'st' in globals():
                st.error(f"❌ {error_msg}")
            raise Exception("PyMuPDF is required for PDF processing")
    
    def pdf_to_images_enhanced(self, pdf_path: str) -> List[Any]:
        """
        Convert ALL PDF pages to high-quality images using PyMuPDF
        Enhanced for better table recognition with 216 DPI (3x zoom)
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List of PIL Image objects
        """
        try:
            from PIL import Image
            doc = fitz.open(pdf_path)
            images = []
            
            logger.info(f"Converting {len(doc)} pages to high-quality images...")
            if 'st' in globals():
                st.info(f"Converting {len(doc)} pages to high-quality images...")
            
            # Progress bar for page conversion
            progress_bar = None
            if 'st' in globals():
                progress_bar = st.progress(0)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Enhanced image quality - 3x zoom = 216 DPI for better text recognition
                mat = fitz.Matrix(3.0, 3.0)  # 3x zoom for 216 DPI
                pix = page.get_pixmap(matrix=mat, alpha=False)  # No alpha for cleaner text
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Convert to RGB for better AI processing
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                images.append(img)
                
                # Update progress
                if progress_bar:
                    progress_bar.progress((page_num + 1) / len(doc))
                
            doc.close()
            logger.info(f"Converted {len(images)} pages using enhanced PyMuPDF (216 DPI)")
            if 'st' in globals():
                st.success(f"✓ Converted {len(images)} pages using enhanced PyMuPDF (216 DPI)")
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            if 'st' in globals():
                st.error(f"Error converting PDF to images: {e}")
            return []
    
    def create_enhanced_table_extraction_prompt(self) -> str:
        """
        Enhanced prompt from model1.py with all specifications for accurate table extraction
        Specialized for financial statements with Quarter & Nine Months format
        
        Returns:
            str: Complete extraction prompt
        """
        return """
        You are an expert PDF table extraction specialist. Analyze this image with extreme precision to extract EXACT data as it appears.

        CRITICAL INSTRUCTIONS FOR ACCURACY:

        STEP 1 - COMPLETE TITLE EXTRACTION:
        - Look for the COMPLETE title including ALL subtitle information
        - Extract titles like: "UNAUDITED CONSOLIDATED FINANCIAL RESULTS FOR THE QUARTER & NINE MONTHS ENDED DECEMBER 31, 2024"
        - ALWAYS include currency/unit information if present: "(Rs. In Lakhs)", "(Rs. In Crores)", etc.
        - Look for text that appears prominently above the table
        - Include any subtitle information that describes the table content
        - Example complete titles:
          * "UNAUDITED CONSOLIDATED FINANCIAL RESULTS FOR THE QUARTER & NINE MONTHS ENDED DECEMBER 31, 2024 (Rs. In Lakhs)"
          * "AUDITED STANDALONE FINANCIAL RESULTS FOR THE QUARTER & YEAR ENDED MARCH 31, 2025 (Rs. In Lakhs)"

        STEP 2 - CRITICAL TEXT RECOGNITION FOR SPECIFIC PATTERNS:
        - MOST IMPORTANT: Look for text patterns that start with a dash/hyphen (-)
        - Specifically look for: "- Deferred Tax Expenses / (Income)"
        - This text appears as an indented sub-item under Tax Expense
        - Extract this EXACTLY as: "- Deferred Tax Expenses / (Income)"
        - DO NOT convert this to "#NAME?" or any error message
        - DO NOT interpret the dash as a negative sign for numbers
        - DO NOT remove the dash or modify the text in any way
        - This is DESCRIPTIVE TEXT, not a formula or calculation

        STEP 3 - QUARTER AND NINE MONTHS COLUMN HANDLING:
        - Look for column headers that contain "Quarter Ended" and "Nine Months Ended"
        - Extract these headers exactly as they appear with dates
        - Examples of expected headers:
          * "Quarter Ended December 31, 2024"
          * "Nine Months Ended December 31, 2024"
          * "Quarter Ended December 31, 2023"
          * "Nine Months Ended December 31, 2023"
        - Preserve the exact format: "Quarter Ended [Date]" and "Nine Months Ended [Date]"
        - Look for additional qualifiers like "Reviewed" or "Unaudited" if present

        STEP 4 - GENERAL TEXT RECOGNITION:
        - Read ALL other text EXACTLY as written in the PDF
        - Pay special attention to negative values in parentheses: (123.45)
        - Look for dash symbols "-" which indicate zero or nil values (different from descriptive text)
        - Preserve all decimal points, commas, and formatting exactly
        - Look carefully at each character to avoid misreading

        STEP 5 - SERIAL NUMBER (Sr. No.) HANDLING:
        - Look for "Sr. No." or "S. No." in the header
        - Serial numbers in this table are: I, II, III, IV, V, VI, VII, VIII, IX (Roman numerals WITHOUT parentheses)
        - Extract EXACTLY as shown:
          * I for first row
          * II for second row  
          * III for third row
          * IV for fourth row
          * V for fifth row
          * VI for sixth row
          * VII for seventh row
          * VIII for eighth row
          * IX for ninth row
        - Do NOT add parentheses if they're not there
        - Do NOT convert to Arabic numbers

        STEP 6 - FINANCIAL DATA HANDLING:
        - Extract ALL numerical values exactly as shown
        - Preserve negative values in parentheses: (135.30), (121.26), (196.58), (552.77)
        - Keep dash symbols as "-" for zero/nil values (when used as data, not as text prefix)
        - Maintain exact decimal precision: 13,542.40, 18,790.26, etc.
        - Include commas in large numbers exactly as shown
        - Do NOT interpret or modify any values

        STEP 7 - COMPLEX TABLE STRUCTURE:
        - Handle multi-level row descriptions correctly
        - For items with sub-items (like "a) Cost of Materials", "b) Purchase"), extract the complete text
        - For indented items like "- Deferred Tax Expenses / (Income)", extract EXACTLY as shown
        - Maintain proper hierarchy and indentation information
        - Extract merged cells and sub-categories properly
        - Remember: Items starting with "- " are descriptive text, not calculations

        STEP 8 - COLUMN HEADERS WITH QUARTER/NINE MONTHS:
        - Extract ALL column headers exactly as shown with the specific format
        - Include headers like:
          * "Quarter Ended December 31, 2024 Reviewed"
          * "Nine Months Ended December 31, 2024 Reviewed"
          * "Quarter Ended December 31, 2023 Reviewed"
          * "Nine Months Ended December 31, 2023 Reviewed"
        - Preserve all header text including qualifiers like "Reviewed", "Unaudited", "Audited"
        - Maintain the exact format: "Quarter Ended [Date]" and "Nine Months Ended [Date]"

        STEP 9 - PRECISE DATA EXTRACTION:
        - Extract ALL visible text from each cell EXACTLY as shown
        - Maintain precise column alignment
        - Include empty cells as empty strings
        - Preserve the exact row and column structure
        - Handle merged cells appropriately
        - Don't modify or interpret the data - extract it exactly
        - NEVER convert valid descriptive text to error messages

        OUTPUT FORMAT - CRITICAL EXAMPLE FOR QUARTER/NINE MONTHS:
        {
            "has_tables": true/false,
            "tables": [
                {
                    "title": "UNAUDITED CONSOLIDATED FINANCIAL RESULTS FOR THE QUARTER & NINE MONTHS ENDED DECEMBER 31, 2024 (Rs. In Lakhs)",
                    "table_number": null,
                    "headers": ["Sr. No.", "Particulars", "Quarter Ended December 31, 2024 Reviewed", "Nine Months Ended December 31, 2024 Reviewed", "Quarter Ended December 31, 2023 Reviewed", "Nine Months Ended December 31, 2023 Reviewed"],
                    "data": [
                        ["I", "Revenue from Operations", "2,369.75", "27,490.52", "2,148.92", "24,117.03"],
                        ["II", "Other Income", "929.74", "1,779.25", "", ""],
                        ["III", "Total Revenue (I+II)", "27,490.52", "63,117.03", "", ""],
                        ["", "Expenses", "", "", "", ""],
                        ["a)", "Cost of Materials Consumed", "15,151.00", "31,781.99", "", ""],
                        ["b)", "Purchase of Traded Goods", "1,721.37", "5,110.47", "", ""],
                        ["c)", "Changes in Inventories of Finished Goods, Work-in-Progress and Stock-in", "829.82", "3,443.16", "", ""],
                        ["IV", "", "", "", "", ""],
                        ["d)", "Employee Benefits Expense", "1,936.50", "3,307.21", "", ""],
                        ["e)", "Manufacturing and Other Expenses", "3,051.79", "7,683.58", "", ""],
                        ["f)", "Finance Cost", "253.80", "554.75", "", ""],
                        ["g)", "Depreciation & Amortisation Expense", "254.70", "665.34", "", ""],
                        ["", "Total Expenses (a to g)", "22,777.28", "52,547.00", "", ""],
                        ["V", "Profit / (Loss) before Exceptional Items and Tax (III-IV)", "4,823.24", "10,570.03", "", ""],
                        ["VI", "Exceptional Items", "-", "-", "", ""],
                        ["VII", "Profit / (Loss) before Tax (V-VI)", "4,823.24", "10,570.03", "", ""],
                        ["VIII", "Tax Expense - Current Tax", "1,109.75", "2,360.00", "", ""],
                        ["", "- Deferred Tax Expenses / (Income)", "(0.52)", "(211.11)", "", ""],
                        ["IX", "Profit / (Loss) for the period (VII-VIII)", "3,632.51", "8,549.21", "", ""]
                    ]
                }
            ]
        }

        CRITICAL ACCURACY REQUIREMENTS:
        1. Include complete title with currency information: "(Rs. In Lakhs)"
        2. Extract Sr. No. as Roman numerals: I, II, III, IV, V, VI, VII, VIII, IX
        3. Preserve negative values in parentheses: (135.30), (121.26)
        4. Keep dash symbols as "-" for nil values in data cells
        5. Extract "- Deferred Tax Expenses / (Income)" EXACTLY as shown - this is descriptive text, not an error
        6. Maintain exact financial formatting with commas
        7. Extract all sub-item descriptions completely
        8. Use "Quarter Ended [Date]" and "Nine Months Ended [Date]" format for column headers
        9. Handle complex table structure with merged cells

        MOST IMPORTANT - AVOID THESE SPECIFIC MISTAKES:
        - Converting "- Deferred Tax Expenses / (Income)" to "#NAME?" or any error message
        - Treating "- Deferred Tax Expenses / (Income)" as a formula or calculation
        - Missing the complete descriptive text that starts with "-"
        - Converting descriptive text starting with "-" to numerical values
        - Not using the proper "Quarter Ended" and "Nine Months Ended" format in headers

        SPECIFIC TEXT PATTERNS TO PRESERVE EXACTLY:
        - "- Deferred Tax Expenses / (Income)" (this is descriptive text, not a calculation)
        - "- Current Tax" (if present)
        - Any other text that starts with "- " (these are descriptions, not formulas)
        - "Quarter Ended [Date]" format for quarterly columns
        - "Nine Months Ended [Date]" format for nine-month columns

        Remember: Text that starts with "- " followed by words is DESCRIPTIVE TEXT that should be extracted exactly as written, never converted to error messages. Column headers should use the exact "Quarter Ended" and "Nine Months Ended" format.
        """
    
    def extract_tables_from_image_enhanced(self, image: Any, page_num: int) -> Dict:
        """
        Extract tables from a single image using Gemini with enhanced specifications
        
        Args:
            image: PIL Image object
            page_num (int): Page number for logging
            
        Returns:
            Dict: Extraction results
        """
        try:
            prompt = self.create_enhanced_table_extraction_prompt()
            
            # Enhanced generation config for better accuracy
            generation_config = {
                'temperature': 0.05,  # Even lower for maximum consistency
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 8192,
            }
            
            logger.info(f"Analyzing page {page_num} with enhanced AI recognition...")
            if 'st' in globals():
                st.info(f"🔍 Analyzing page {page_num} with enhanced AI recognition...")
            
            response = self.model.generate_content(
                [prompt, image],
                generation_config=generation_config
            )
            
            # Enhanced JSON parsing with multiple fallback methods
            response_text = response.text.strip()
            
            # Clean up response text
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            elif response_text.startswith('json'):
                response_text = response_text[4:].strip()
            
            if response_text.endswith('```'):
                response_text = response_text[:-3].strip()
            
            try:
                result = json.loads(response_text)
                
                # Enhanced validation
                if not isinstance(result, dict):
                    logger.warning(f"Page {page_num}: Invalid response format")
                    if 'st' in globals():
                        st.warning(f"Page {page_num}: Invalid response format")
                    return {"has_tables": False, "tables": []}
                
                if "has_tables" not in result:
                    logger.warning(f"Page {page_num}: Missing 'has_tables' key")
                    if 'st' in globals():
                        st.warning(f"Page {page_num}: Missing 'has_tables' key")
                    return {"has_tables": False, "tables": []}
                
                if result.get("has_tables") and "tables" not in result:
                    logger.warning(f"Page {page_num}: Missing 'tables' key")
                    if 'st' in globals():
                        st.warning(f"Page {page_num}: Missing 'tables' key")
                    return {"has_tables": False, "tables": []}
                
                # Enhanced table validation
                if result.get("has_tables") and result.get("tables"):
                    valid_tables = []
                    for i, table in enumerate(result["tables"]):
                        if not isinstance(table, dict):
                            continue
                        
                        # Ensure required keys exist
                        if "headers" not in table:
                            table["headers"] = []
                        if "data" not in table:
                            table["data"] = []
                        if "title" not in table:
                            table["title"] = None
                        
                        # Check for specific patterns
                        has_deferred_tax = any(
                            any("- Deferred Tax" in str(cell) for cell in row) 
                            for row in table.get("data", [])
                        )
                        
                        if has_deferred_tax:
                            logger.info(f"Page {page_num}: Detected '- Deferred Tax Expenses' pattern correctly")
                            if 'st' in globals():
                                st.success(f"✓ Page {page_num}: Detected '- Deferred Tax Expenses' pattern correctly")
                        
                        # Check for Quarter/Nine Months format
                        headers = table.get("headers", [])
                        has_quarter_format = any(
                            "Quarter Ended" in str(header) or "Nine Months Ended" in str(header)
                            for header in headers
                        )
                        
                        if has_quarter_format:
                            logger.info(f"Page {page_num}: Detected Quarter/Nine Months format correctly")
                            if 'st' in globals():
                                st.success(f"✓ Page {page_num}: Detected Quarter/Nine Months format correctly")
                        
                        valid_tables.append(table)
                    
                    result["tables"] = valid_tables
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Page {page_num}: JSON parsing error - {e}")
                if 'st' in globals():
                    st.error(f"Page {page_num}: JSON parsing error - {e}")
                return {"has_tables": False, "tables": []}
                
        except Exception as e:
            logger.error(f"Page {page_num}: Error extracting tables - {e}")
            if 'st' in globals():
                st.error(f"Page {page_num}: Error extracting tables - {e}")
            return {"has_tables": False, "tables": []}
    
    def normalize_title_for_grouping_enhanced(self, title: str, page_num: int) -> str:
        """
        Enhanced title normalization for better continuation detection
        
        Args:
            title (str): Original title
            page_num (int): Page number
            
        Returns:
            str: Normalized title for grouping
        """
        if not title or title.strip() == '':
            return f"Table_Page_{page_num}"
        
        normalized = re.sub(r'\s+', ' ', title.strip())
        
        # Enhanced continuation patterns
        continuation_patterns = [
            r'\s*\(continued\)',
            r'\s*\(contd\.?\)',
            r'\s*\(cont\.?\)',
            r'\s*continued',
            r'\s*contd\.?',
            r'\s*\-\s*continued',
            r'\s*\-\s*contd\.?',
            r'page\s*\d+',
            r'sheet\s*\d+',
            r'\s*\-\s*page\s*\d+'
        ]
        
        for pattern in continuation_patterns:
            normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
        
        # Enhanced company and format patterns
        enhanced_patterns = [
            (r'HDFC\s+Life\s+Insurance\s+Company\s+Limited?', 'HDFC Life Insurance Company Limited'),
            (r'LLOYDS\s+ENGINEERING\s+WORKS\s+LIMITED', 'LLOYDS ENGINEERING WORKS LIMITED'),
            (r'UNAUDITED\s+CONSOLIDATED\s+FINANCIAL\s+RESULTS', 'UNAUDITED CONSOLIDATED FINANCIAL RESULTS'),
            (r'AUDITED\s+STANDALONE\s+FINANCIAL\s+RESULTS', 'AUDITED STANDALONE FINANCIAL RESULTS'),
            (r'for\s+the\s+Quarter\s+&\s+Nine\s+Months\s+ended', 'for the Quarter & Nine Months ended'),
            (r'for\s+the\s+Quarter\s+&\s+Year\s+ended', 'for the Quarter & Year ended'),
            (r'December\s+31,?\s*2024', 'December 31, 2024'),
            (r'March\s+31,?\s*2025', 'March 31, 2025'),
            (r'Rs\.?\s*[Ii]n\s*Lakhs?', 'Rs. In Lakhs'),
            (r'₹\s*[Ii]n\s*Lakhs?', '₹ In Lakhs')
        ]
        
        for pattern, replacement in enhanced_patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return normalized.strip()
    
    def are_headers_compatible_enhanced(self, headers1: List, headers2: List) -> bool:
        """
        Enhanced header compatibility check for continuation detection
        
        Args:
            headers1 (List): First set of headers
            headers2 (List): Second set of headers
            
        Returns:
            bool: True if headers are compatible
        """
        if not headers1 or not headers2:
            return True
        
        # Normalize headers
        norm_headers1 = [re.sub(r'\s+', ' ', str(h).strip().lower()) for h in headers1]
        norm_headers2 = [re.sub(r'\s+', ' ', str(h).strip().lower()) for h in headers2]
        
        # Exact match
        if norm_headers1 == norm_headers2:
            return True
        
        # Enhanced similarity check
        if len(norm_headers1) > 0 and len(norm_headers2) > 0:
            common_headers = set(norm_headers1) & set(norm_headers2)
            max_len = max(len(norm_headers1), len(norm_headers2))
            overlap_ratio = len(common_headers) / max_len
            
            if overlap_ratio >= 0.7:  # 70% similarity
                return True
        
        # Check for financial statement patterns
        financial_keywords = [
            'particulars', 'sr. no', 'quarter ended', 'nine months ended',
            'march', 'december', 'audited', 'reviewed'
        ]
        
        headers1_financial = any(keyword in ' '.join(norm_headers1) for keyword in financial_keywords)
        headers2_financial = any(keyword in ' '.join(norm_headers2) for keyword in financial_keywords)
        
        return headers1_financial and headers2_financial
    
    def save_enhanced_table_to_csv(self, combined_table: Dict, pdf_name: str) -> Optional[str]:
        """
        Save table with enhanced Excel compatibility and formula fixes
        Uses extracted table title for filename instead of generic names
        
        Args:
            combined_table (Dict): Combined table data
            pdf_name (str): PDF filename
            
        Returns:
            Optional[str]: Path to saved CSV file
        """
        try:
            title = combined_table.get('title', '')
            if title:
                # Enhanced filename cleaning using actual table title
                safe_filename = re.sub(r'[<>:"/\\|?*]', '', title)
                safe_filename = safe_filename.replace('(Rs. In Lakhs)', '').strip()
                safe_filename = safe_filename.replace('(Rs. In Crores)', '').strip()
                safe_filename = re.sub(r'\s+', ' ', safe_filename)
                # Limit filename length but preserve meaningful content
                if len(safe_filename) > 80:
                    safe_filename = safe_filename[:80].strip()
                filename = f"{safe_filename}.csv"
            else:
                # Only use fallback if no title is available
                filename = f"Table_from_Page_{combined_table.get('pages', ['Unknown'])[0]}.csv"
            
            filepath = self.output_dir / filename
            
            headers = combined_table.get('headers', [])
            data = combined_table.get('data', [])
            
            if not data:
                logger.warning(f"No data found in table: {title}")
                if 'st' in globals():
                    st.warning(f"No data found in table: {title}")
                return None
            
            # Enhanced Excel formula fix function
            def fix_excel_formula_issues_enhanced(cell_value):
                """Enhanced fix for Excel formula interpretation issues"""
                if isinstance(cell_value, str):
                    # Critical fix for "- Deferred Tax Expenses / (Income)"
                    if cell_value.startswith('-') and any(c.isalpha() for c in cell_value):
                        return f"'{cell_value}"
                    # Fix for other formula-like patterns
                    elif cell_value.startswith(('=', '+', '@')):
                        return f"'{cell_value}"
                return cell_value
            
            # Apply enhanced fixes
            fixed_data = []
            deferred_tax_fixed = 0
            
            for row in data:
                fixed_row = []
                for cell in row:
                    original_cell = cell
                    fixed_cell = fix_excel_formula_issues_enhanced(cell)
                    
                    # Count deferred tax fixes
                    if (isinstance(original_cell, str) and 
                        "- Deferred Tax" in original_cell and 
                        fixed_cell != original_cell):
                        deferred_tax_fixed += 1
                    
                    fixed_row.append(fixed_cell)
                fixed_data.append(fixed_row)
            
            if deferred_tax_fixed > 0:
                logger.info(f"Fixed {deferred_tax_fixed} '- Deferred Tax' entries for Excel compatibility")
                if 'st' in globals():
                    st.success(f"✅ Fixed {deferred_tax_fixed} '- Deferred Tax' entries for Excel compatibility")
            
            # Enhanced column alignment
            if headers and fixed_data:
                max_data_cols = max(len(row) for row in fixed_data) if fixed_data else 0
                
                if len(headers) != max_data_cols:
                    logger.info(f"Adjusting columns: {len(headers)} headers → {max_data_cols} data columns")
                    if 'st' in globals():
                        st.info(f"Adjusting columns: {len(headers)} headers → {max_data_cols} data columns")
                    
                    if len(headers) < max_data_cols:
                        for i in range(len(headers), max_data_cols):
                            headers.append(f"Column_{i+1}")
                
                # Ensure consistent row lengths
                adjusted_data = []
                for row in fixed_data:
                    if len(row) < len(headers):
                        adjusted_row = row + [''] * (len(headers) - len(row))
                    else:
                        adjusted_row = row[:len(headers)]
                    adjusted_data.append(adjusted_row)
                
                df = pd.DataFrame(adjusted_data, columns=headers)
            else:
                logger.warning("No headers or data found")
                if 'st' in globals():
                    st.warning("No headers or data found")
                return None
            
            # Save with enhanced formatting
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                # Add title
                if title:
                    csvfile.write(f'"{title}"\n\n')
                
                # Add page information
                pages = combined_table.get('pages', [])
                if len(pages) > 1:
                    csvfile.write(f'"Combined from pages: {", ".join(map(str, pages))}"\n\n')
                
                # Write DataFrame
                df.to_csv(csvfile, index=False)
            
            logger.info(f"Saved: {filepath.name}")
            if 'st' in globals():
                st.success(f"✅ Saved: {filepath.name}")
                st.info(f"   Size: {len(df)} rows × {len(df.columns)} columns")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving table: {e}")
            if 'st' in globals():
                st.error(f"Error saving table: {e}")
            return None
    
    def generate_enhanced_summary_report(self, results: Dict) -> str:
        """
        Generate comprehensive summary report
        
        Args:
            results (Dict): Processing results
            
        Returns:
            str: Path to summary report file
        """
        report_path = self.output_dir / f"{results['pdf_name']}_ENHANCED_extraction_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ENHANCED PDF TABLE EXTRACTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"PDF File: {results['pdf_name']}\n")
            f.write(f"Output Directory: {results['output_directory']}\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("EXTRACTION STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Pages Processed: {results['total_pages']}\n")
            f.write(f"Pages with Tables: {results['pages_with_tables']}\n")
            f.write(f"Total Tables Extracted: {results['total_tables_extracted']}\n")
            f.write(f"CSV Files Created: {len(results['csv_files'])}\n\n")
            
            # Processing details
            details = results.get('processing_details', {})
            f.write("PROCESSING DETAILS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"AI Model: Gemini 2.0 Flash\n")
            f.write(f"Title Extraction: {details.get('title_extraction_method', 'N/A')}\n")
            f.write(f"Image Quality: {details.get('image_conversion', 'N/A')}\n")
            f.write(f"Continuation Detection: {details.get('continuation_detection', 'N/A')}\n")
            f.write(f"Excel Formula Fixes: {details.get('excel_formula_fixes', 'N/A')}\n")
            f.write(f"Roman Numerals Found: {'✓' if details.get('roman_numerals_preserved') else '✗'}\n")
            f.write(f"Quarter/Nine Months Format: {'✓' if details.get('quarter_nine_months_format') else '✗'}\n")
            f.write(f"Deferred Tax Entries: {'✓' if details.get('deferred_tax_entries_found') else '✗'}\n\n")
            
            # Extracted titles
            if results.get('extracted_titles'):
                f.write("EXTRACTED TITLES:\n")
                f.write("-" * 30 + "\n")
                for i, title in enumerate(results['extracted_titles'], 1):
                    f.write(f"{i}. {title}\n")
                f.write("\n")
            
            # CSV files
            f.write("GENERATED CSV FILES:\n")
            f.write("-" * 30 + "\n")
            for csv_file in results['csv_files']:
                f.write(f"• {Path(csv_file).name}\n")
            f.write("\n")
            
            # Page details
            f.write("PAGE-BY-PAGE ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            for page_result in results['page_results']:
                f.write(f"Page {page_result['page_number']}: ")
                if page_result['has_tables']:
                    f.write(f"{page_result['tables_count']} table(s)\n")
                    for table in page_result.get('tables', []):
                        f.write(f"  - {table.get('title', 'Untitled')} ")
                        f.write(f"({table.get('rows', 0)} rows, {table.get('columns', 0)} cols)")
                        
                        # Special features
                        features = []
                        if table.get('has_deferred_tax'):
                            features.append("Deferred Tax")
                        if table.get('has_quarter_format'):
                            features.append("Q&9M Format")
                        if table.get('has_roman_numerals'):
                            features.append("Roman Numerals")
                        
                        if features:
                            f.write(f" [Features: {', '.join(features)}]")
                        f.write("\n")
                else:
                    f.write("No tables\n")
        
        logger.info(f"Summary report saved: {report_path.name}")
        if 'st' in globals():
            st.success(f"📋 Summary report saved: {report_path.name}")
        return str(report_path)
    
    def process_all_pages_enhanced(self, pdf_path: str) -> Dict:
        """
        Process ALL pages of PDF with enhanced table extraction and continuation detection
        Main processing function that orchestrates the entire extraction workflow
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict: Complete processing results
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Setup output directory
        self.setup_output_directory(str(pdf_path))
        
        pdf_name = pdf_path.stem
        logger.info(f"Starting enhanced processing of: {pdf_name}")
        if 'st' in globals():
            st.info(f"🚀 Starting enhanced processing of: {pdf_name}")
        
        # Convert ALL pages to images
        images = self.pdf_to_images_enhanced(str(pdf_path))
        if not images:
            return {
                "error": "Failed to convert PDF to images",
                "pdf_name": pdf_name,
                "total_pages": 0,
                "pages_with_tables": 0,
                "total_tables_extracted": 0,
                "csv_files": [],
                "page_results": []
            }
        
        results = {
            "pdf_name": pdf_name,
            "output_directory": str(self.output_dir),
            "total_pages": len(images),
            "pages_with_tables": 0,
            "total_tables_extracted": 0,
            "csv_files": [],
            "page_results": [],
            "extracted_titles": [],
            "processing_details": {
                "ai_model": "Gemini 2.0 Flash",
                "title_extraction_method": "Enhanced PDF metadata + first page analysis",
                "image_conversion": "PyMuPDF with 3x zoom (216 DPI)",
                "continuation_detection": "Enhanced title normalization with financial patterns",
                "excel_formula_fixes": "Applied to prevent #NAME? errors for '- Deferred Tax'",
                "roman_numerals_preserved": False,
                "quarter_nine_months_format": False,
                "deferred_tax_entries_found": False
            }
        }
        
        # Enhanced table grouping by title
        tables_by_title = {}
        
        # Process each page with enhanced analysis
        total_pages = len(images)
        main_progress = None
        if 'st' in globals():
            main_progress = st.progress(0)
        
        for page_num, image in enumerate(images, 1):
            logger.info(f"Processing Page {page_num}/{total_pages}")
            if 'st' in globals():
                st.subheader(f"📄 Processing Page {page_num}/{total_pages}")
            
            try:
                # Extract tables from current page
                extraction_result = self.extract_tables_from_image_enhanced(image, page_num)
                
                page_result = {
                    "page_number": page_num,
                    "has_tables": extraction_result.get("has_tables", False),
                    "tables_count": len(extraction_result.get("tables", [])),
                    "tables": []
                }
                
                if extraction_result.get("has_tables", False):
                    results["pages_with_tables"] += 1
                    tables = extraction_result.get("tables", [])
                    
                    for table_num, table_data in enumerate(tables, 1):
                        title = table_data.get('title', 'Untitled Table')
                        logger.info(f"Found table: {title}")
                        if 'st' in globals():
                            st.success(f"✅ Found table: {title}")
                        
                        # Track extracted titles
                        if table_data.get('title'):
                            results["extracted_titles"].append(table_data.get('title'))
                        
                        # Check for specific patterns
                        data = table_data.get('data', [])
                        headers = table_data.get('headers', [])
                        
                        # Check for Roman numerals
                        has_roman = any(
                            any(str(cell).strip() in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX'] 
                                for cell in row) for row in data
                        )
                        if has_roman:
                            results["processing_details"]["roman_numerals_preserved"] = True
                            logger.info("Roman numerals detected and preserved")
                            if 'st' in globals():
                                st.success("✅ Roman numerals detected and preserved")
                        
                        # Check for Quarter/Nine Months format
                        has_quarter_format = any(
                            "Quarter Ended" in str(header) or "Nine Months Ended" in str(header)
                            for header in headers
                        )
                        if has_quarter_format:
                            results["processing_details"]["quarter_nine_months_format"] = True
                            logger.info("Quarter & Nine Months format detected")
                            if 'st' in globals():
                                st.success("✅ Quarter & Nine Months format detected")
                        
                        # Check for deferred tax entries
                        has_deferred_tax = any(
                            any("- Deferred Tax" in str(cell) for cell in row) 
                            for row in data
                        )
                        if has_deferred_tax:
                            results["processing_details"]["deferred_tax_entries_found"] = True
                            logger.info("Deferred Tax entries found and preserved correctly")
                            if 'st' in globals():
                                st.success("✅ Deferred Tax entries found and preserved correctly")
                        
                        # Enhanced title normalization
                        normalized_title = self.normalize_title_for_grouping_enhanced(title, page_num)
                        
                        # Group tables by normalized title for continuation handling
                        if normalized_title not in tables_by_title:
                            tables_by_title[normalized_title] = {
                                "title": title,
                                "headers": table_data.get('headers', []),
                                "data": table_data.get('data', []),
                                "pages": [page_num],
                                "table_numbers": [table_num],
                                "original_titles": [title]
                            }
                            logger.info(f"Created new table group: {normalized_title}")
                            if 'st' in globals():
                                st.info(f"📝 Created new table group: {normalized_title}")
                        else:
                            # Check for continuation compatibility
                            existing_table = tables_by_title[normalized_title]
                            
                            if self.are_headers_compatible_enhanced(existing_table["headers"], table_data.get('headers', [])):
                                existing_table["data"].extend(table_data.get('data', []))
                                existing_table["pages"].append(page_num)
                                existing_table["table_numbers"].append(table_num)
                                existing_table["original_titles"].append(title)
                                logger.info(f"Combined continuation data from page {page_num}")
                                if 'st' in globals():
                                    st.success(f"🔗 Combined continuation data from page {page_num}")
                            else:
                                # Different structure - create variant
                                alt_title = f"{normalized_title}_v{len([k for k in tables_by_title.keys() if k.startswith(normalized_title)])+1}"
                                tables_by_title[alt_title] = {
                                    "title": title,
                                    "headers": table_data.get('headers', []),
                                    "data": table_data.get('data', []),
                                    "pages": [page_num],
                                    "table_numbers": [table_num],
                                    "original_titles": [title]
                                }
                                logger.info(f"Created variant table: {alt_title}")
                                if 'st' in globals():
                                    st.info(f"📝 Created variant table: {alt_title}")
                        
                        # Add to page results
                        page_result["tables"].append({
                            "title": table_data.get("title"),
                            "table_number": table_data.get("table_number"),
                            "normalized_title": normalized_title,
                            "rows": len(table_data.get("data", [])),
                            "columns": len(table_data.get("headers", [])),
                            "has_deferred_tax": has_deferred_tax,
                            "has_quarter_format": has_quarter_format,
                            "has_roman_numerals": has_roman
                        })
                else:
                    logger.info(f"No tables found on page {page_num}")
                    if 'st' in globals():
                        st.info(f"ℹ️ No tables found on page {page_num}")
                
                results["page_results"].append(page_result)
                
                # Update progress
                if main_progress:
                    main_progress.progress(page_num / total_pages)
                
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                if 'st' in globals():
                    st.error(f"❌ Error processing page {page_num}: {e}")
                page_result = {
                    "page_number": page_num,
                    "has_tables": False,
                    "tables_count": 0,
                    "tables": [],
                    "error": str(e)
                }
                results["page_results"].append(page_result)
        
        # Save combined tables with enhanced processing
        logger.info("Saving Combined Tables")
        if 'st' in globals():
            st.subheader("💾 Saving Combined Tables")
            
        for normalized_title, combined_table in tables_by_title.items():
            logger.info(f"Saving: {normalized_title}")
            if 'st' in globals():
                st.info(f"Saving: {normalized_title}")
                st.info(f"Pages: {combined_table['pages']} | Rows: {len(combined_table['data'])}")
            
            csv_path = self.save_enhanced_table_to_csv(combined_table, pdf_name)
            
            if csv_path:
                results["csv_files"].append(csv_path)
                results["total_tables_extracted"] += 1
        
        # Generate enhanced summary
        summary_path = self.generate_enhanced_summary_report(results)
        
        return results


# Streamlit Web Interface
def main():
    """
    Main Streamlit application function
    """
    st.set_page_config(
        page_title="Enhanced PDF Table Extractor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Enhanced PDF Table Extractor")
    st.markdown("### Extract tables from ALL pages with AI-powered precision")
    st.markdown("**Powered by Google Gemini 2.0 Flash with enhanced financial statement recognition**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            help="Get your API key from Google AI Studio"
        )
        
        st.markdown("[Get API Key](https://makersuite.google.com/app/apikey)")
        
        if api_key:
            st.success("✅ API Key provided")
        
        st.header("📋 Features")
        st.markdown("""
        **Enhanced Extraction:**
        - ✅ ALL pages processed
        - ✅ High-quality 216 DPI conversion
        - ✅ Quarter & Nine Months format
        - ✅ Roman numerals preserved
        - ✅ Deferred Tax entries fixed
        - ✅ Continuation table detection
        - ✅ Excel formula compatibility
        - ✅ Table title-based filenames
        """)
        
        st.header("🎯 Optimized For")
        st.markdown("""
        - Financial statements
        - Quarterly reports
        - Annual reports
        - Consolidated results
        - Audited/Unaudited statements
        """)
        
        st.header("🆕 Updates")
        st.markdown("""
        - **Gemini 2.0 Flash:** Latest AI model
        - **Smart Filenames:** Uses extracted table titles
        - **Individual Downloads:** Separate file for each table
        """)
    
    # Main interface
    uploaded_file = st.file_uploader(
        "📁 Upload PDF File",
        type="pdf",
        help="Upload a PDF containing tables (financial statements, reports, etc.)"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # File details
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{uploaded_file.size / (1024*1024):.2f} MB")
        with col2:
            st.metric("File Type", "PDF")
        with col3:
            st.metric("Status", "Ready")
        
        # Process button
        if st.button("🚀 Extract Tables from ALL Pages", type="primary", disabled=not api_key):
            if not api_key:
                st.error("❌ Please provide your Gemini API key in the sidebar")
                return
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            try:
                # Initialize extractor
                extractor = EnhancedPDFTableExtractor(api_key)
                
                # Process with enhanced extraction
                st.info("🔄 Starting enhanced table extraction with Gemini 2.0 Flash...")
                with st.spinner("Processing all pages..."):
                    results = extractor.process_all_pages_enhanced(tmp_file_path)
                
                # Display results
                if "error" in results:
                    st.error(f"❌ {results['error']}")
                    return
                
                # Success metrics
                st.success("🎉 Extraction completed successfully!")
                
                # Results dashboard
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📄 Total Pages", results['total_pages'])
                with col2:
                    st.metric("📊 Pages with Tables", results['pages_with_tables'])
                with col3:
                    st.metric("📋 Tables Extracted", results['total_tables_extracted'])
                with col4:
                    st.metric("💾 CSV Files", len(results['csv_files']))
                
                # Processing details
                details = results.get('processing_details', {})
                st.subheader("🔍 Processing Analysis")
                
                detail_cols = st.columns(4)
                with detail_cols[0]:
                    st.info(f"**AI Model:** {details.get('ai_model', 'Gemini 2.0 Flash')}")
                with detail_cols[1]:
                    st.info(f"**Roman Numerals:** {'✅ Found' if details.get('roman_numerals_preserved') else '❌ Not found'}")
                with detail_cols[2]:
                    st.info(f"**Q&9M Format:** {'✅ Detected' if details.get('quarter_nine_months_format') else '❌ Not found'}")
                with detail_cols[3]:
                    st.info(f"**Deferred Tax:** {'✅ Found & Fixed' if details.get('deferred_tax_entries_found') else '❌ Not found'}")
                
                # CSV files with individual downloads
                if results['csv_files']:
                    st.subheader("💾 Individual Table Downloads")
                    st.markdown("**Each table saved with its extracted title as filename:**")
                    
                    # Create a container that persists after downloads
                    download_container = st.container()
                    
                    with download_container:
                        # Create download section for each CSV file
                        for i, csv_path in enumerate(results['csv_files'], 1):
                            if os.path.exists(csv_path):
                                filename = os.path.basename(csv_path)
                                
                                # Read file content for download
                                with open(csv_path, 'r', encoding='utf-8') as f:
                                    csv_content = f.read()
                                
                                # Get file size
                                file_size = len(csv_content.encode('utf-8'))
                                
                                # Create columns for better layout
                                col1, col2, col3 = st.columns([4, 1, 1])
                                
                                with col1:
                                    st.write(f"**{i}. {filename}**")
                                with col2:
                                    st.write(f"{file_size / 1024:.1f} KB")
                                with col3:
                                    # Create a unique key that includes current time to avoid conflicts
                                    unique_key = f"csv_download_{i}_{abs(hash(filename))}"
                                    
                                    # Use a form to prevent auto-rerun
                                    with st.form(key=f"form_{unique_key}"):
                                        submitted = st.form_submit_button("📥 Download", use_container_width=True)
                                        if submitted:
                                            # Create download using st.download_button outside form
                                            st.success(f"✅ Preparing download for {filename}")
                                    
                                    # Always show download button (this approach keeps it persistent)
                                    st.download_button(
                                        label="📥 Get File",
                                        data=csv_content.encode('utf-8'),
                                        file_name=filename,
                                        mime="text/csv",
                                        key=f"persistent_download_{i}_{abs(hash(filename))}",
                                        help=f"Click to download {filename}"
                                    )
                    
                    st.divider()
                    
                    # Bulk download as ZIP
                    st.subheader("📦 Bulk Download")
                    
                    # Create download zip
                    zip_buffer = io.BytesIO()
                    files_added = 0
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for csv_path in results['csv_files']:
                            if os.path.exists(csv_path):
                                zip_file.write(csv_path, os.path.basename(csv_path))
                                files_added += 1
                    
                    if files_added > 0:
                        zip_buffer.seek(0)
                        
                        # Download button for all files
                        st.download_button(
                            label=f"📦 Download All {files_added} CSV Files (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name=f"{results['pdf_name']}_extracted_tables.zip",
                            mime="application/zip",
                            key="bulk_zip_download"
                        )
                    else:
                        st.error("❌ No files available for bulk download")
                
                # Page-by-page results
                st.subheader("📄 Page-by-Page Analysis")
                for page_result in results['page_results']:
                    with st.expander(f"Page {page_result['page_number']}" + 
                                   (f" - {page_result['tables_count']} table(s)" if page_result['has_tables'] else " - No tables")):
                        
                        if page_result['has_tables']:
                            for table in page_result.get('tables', []):
                                st.write(f"**Title:** {table.get('title', 'Untitled')}")
                                
                                # Table features
                                features = []
                                if table.get('has_deferred_tax'):
                                    features.append("🔸 Deferred Tax Entries")
                                if table.get('has_quarter_format'):
                                    features.append("📅 Quarter/Nine Months Format")
                                if table.get('has_roman_numerals'):
                                    features.append("🔢 Roman Numerals")
                                
                                if features:
                                    st.write("**Features:** " + " | ".join(features))
                                
                                st.write(f"**Size:** {table.get('rows', 0)} rows × {table.get('columns', 0)} columns")
                                st.divider()
                        else:
                            st.write("No tables detected on this page")
                
                # Extracted titles
                if results.get('extracted_titles'):
                    st.subheader("📝 Extracted Table Titles")
                    for i, title in enumerate(results['extracted_titles'], 1):
                        st.write(f"{i}. {title}")
                
                # Output directory info
                st.info(f"📁 **Output Directory:** {results['output_directory']}")
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)
                logger.error(f"Error during processing: {str(e)}")
                logger.error(traceback.format_exc())
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
    
    # Instructions and deployment info
    st.divider()
    
    with st.expander("📖 Instructions & Usage Guide"):
        st.markdown("""
        ## How to Use
        1. **Get API Key:** Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. **Upload PDF:** Choose a PDF file containing tables
        3. **Extract:** Click the extraction button to process all pages
        4. **Download:** Get individual CSV files with meaningful filenames based on table titles
        
        ## Key Features
        - **All Pages Processing:** Every page is analyzed for tables
        - **Enhanced Recognition:** Specialized for financial statements
        - **Continuation Detection:** Automatically combines split tables
        - **Excel Compatibility:** Fixes formula interpretation issues
        - **Quarter/Nine Months:** Properly handles Q&9M financial formats
        - **Roman Numerals:** Preserves Sr. No. formatting (I, II, III, etc.)
        - **Deferred Tax Fix:** Prevents "- Deferred Tax Expenses" from becoming #NAME?
        - **Smart Filenames:** Uses extracted table titles instead of generic names
        - **Individual Downloads:** Separate download for each table
        
        ## Latest Updates
        - **Gemini 2.0 Flash:** Using the latest AI model for better accuracy
        - **Table Title Filenames:** CSV files named using actual extracted table titles
        - **No Generic Names:** Eliminates files like "tmpot0ijrtd_Combined_Table.csv"
        
        ## Supported Table Types
        - Financial statements (P&L, Balance Sheet, Cash Flow)
        - Quarterly and annual reports
        - Consolidated and standalone results
        - Notes to financial statements
        - Complex multi-page tables
        
        ## Technical Specifications
        - **AI Model:** Google Gemini 2.0 Flash (Latest)
        - **Image Quality:** 216 DPI (3x zoom)
        - **Processing:** All pages analyzed
        - **Output:** CSV files with Excel compatibility and meaningful filenames
        - **Special Handling:** Financial patterns, Roman numerals, continuation tables
        
        ## File Naming Strategy
        - **Smart Naming:** Uses extracted table titles for filenames
        - **Clean Names:** Removes special characters and currency info from filenames
        - **Unique Files:** Each table gets its own CSV file
        - **No Temporary Names:** No more generic "tmpXXX_Combined_Table.csv" files
        """)


if __name__ == "__main__":
    main()
