# Enterprise Dataset Preparation Guide

## 📊 Complete Guide to Preparing Training Data

### Overview

This guide explains how to prepare, format, and organize enterprise data for training your Small Language Model. The quality and quantity of your training data directly impacts model performance.

---

## Part 1: Data Requirements

### Minimum Data Requirements

| Model Quality | Documents | Text Volume | Training Time |
|--------------|-----------|-------------|---------------|
| Basic (POC)  | 100+      | 100K tokens | 30 mins - 1 hour |
| Functional   | 1,000+    | 1M tokens   | 2-4 hours |
| Good         | 10,000+   | 10M tokens  | 8-12 hours |
| Production   | 50,000+   | 50M+ tokens | 1-3 days |

**Recommendation**: Start with 1,000+ documents for a functional model, then scale up.

### Data Quality Checklist

✅ **Good Quality Data:**
- Clean, well-formatted text
- Relevant to your domain
- Diverse document types
- Minimal duplicates
- Proper grammar and spelling
- Clear structure

❌ **Poor Quality Data:**
- Heavily corrupted PDFs
- Excessive boilerplate text
- Pure tables without context
- Low-quality OCR results
- Machine-translated gibberish
- Extreme duplication

---

## Part 2: Supported Data Sources

### 2.1 SharePoint / OneDrive

**What to Export:**
- Documents (.docx, .doc)
- PDFs (.pdf)
- Presentations (.pptx)
- Spreadsheets (.xlsx) - with text content
- Text files (.txt)

**How to Export:**

**Option A: Manual Download (Small Scale)**
1. Navigate to your SharePoint site
2. Select documents → Download
3. Extract to `data/raw/sharepoint/`

**Option B: Using SharePoint API (Large Scale)**
```python
# Example using Microsoft Graph API
# Requires: pip install msal requests

from msal import PublicClientApplication
import requests

# Configure authentication
app = PublicClientApplication(
    client_id="YOUR_CLIENT_ID",
    authority="https://login.microsoftonline.com/YOUR_TENANT_ID"
)

# Get token
result = app.acquire_token_interactive(scopes=["Files.Read.All"])
token = result["access_token"]

# Download files
headers = {"Authorization": f"Bearer {token}"}
# ... download files logic
```

**Best Practices:**
- Only include business-relevant documents
- Exclude system files and drafts
- Organize by department or topic
- Remove outdated documents

### 2.2 Google Drive

**What to Export:**
- Google Docs → Convert to .docx or .txt
- PDFs
- Sheets → Export as .xlsx or .csv
- Slides → Export as .pptx

**How to Export:**

**Option A: Google Takeout (Entire Drive)**
1. Go to [takeout.google.com](https://takeout.google.com)
2. Select "Drive"
3. Choose format (recommended: docx for Docs, xlsx for Sheets)
4. Create export
5. Download and extract to `data/raw/google_drive/`

**Option B: Selective Download**
1. Open Google Drive
2. Select documents
3. Right-click → Download
4. Place in `data/raw/google_drive/`

**Option C: Using Google Drive API**
```python
# Example using Google Drive API
# Requires: pip install google-auth google-auth-oauthlib google-api-python-client

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# Authenticate and create service
service = build('drive', 'v3', credentials=creds)

# List files
results = service.files().list(
    pageSize=100,
    fields="files(id, name, mimeType)"
).execute()

# Download files
# ... download logic
```

**Best Practices:**
- Convert Google Docs to native formats
- Include folder structure in filenames
- Remove duplicate versions
- Exclude multimedia files

### 2.3 Slack

**What to Export:**
- Channel messages
- Direct messages (if permitted)
- Thread conversations
- Shared documents

**How to Export:**

**Workspace Export (Recommended)**
1. Go to Workspace Settings (must be admin)
2. Click "Import/Export Data"
3. Select "Export" tab
4. Choose date range
5. Click "Start Export"
6. Download ZIP file when ready
7. Extract to `data/raw/slack/`

**File Structure:**
```
slack/
├── channel1/
│   └── 2024-01-01.json
├── channel2/
│   └── 2024-01-01.json
└── users.json
```

**Message Format:**
```json
[
  {
    "type": "message",
    "user": "U123ABC",
    "text": "Has anyone tested the new feature?",
    "ts": "1234567890.123456"
  }
]
```

**Best Practices:**
- Exclude bot messages (optional)
- Remove private/sensitive channels
- Filter by date range for relevance
- Consider message context/threads

### 2.4 Microsoft Teams

**What to Export:**
- Chat messages
- Channel posts
- Shared files
- Meeting notes

**How to Export:**

**Option A: Manual Export**
1. Open Teams
2. Go to each channel
3. Download shared files
4. Copy important messages to text files

**Option B: Using Microsoft Graph API**
```python
# Requires Microsoft Graph API access
# Get team messages
endpoint = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages"
# ... API calls
```

**Best Practices:**
- Focus on business-critical channels
- Include meeting notes and summaries
- Preserve thread context
- Exclude system notifications

### 2.5 Fireflies.ai (Meeting Transcripts)

**What to Export:**
- Meeting transcripts
- Summary notes
- Action items

**How to Export:**

1. Log into Fireflies.ai
2. Go to Notebooks
3. Select meetings
4. Click Export
5. Choose format: Text or JSON
6. Download to `data/raw/fireflies/`

**Transcript Format:**
```
Meeting: Product Planning Call
Date: 2024-01-15
Duration: 45 minutes

Transcript:
[00:00] John: Let's discuss Q2 roadmap...
[02:15] Sarah: I suggest we prioritize mobile...
[05:30] Mike: What about the API updates?...
```

**Best Practices:**
- Include meeting context (title, date)
- Remove speaker identification if privacy required
- Clean up transcription errors
- Organize by topic/department

### 2.6 GitHub Repositories

**What to Include:**
- Source code files
- Documentation (README.md, docs/)
- Configuration files
- Comments and docstrings

**How to Export:**

```bash
# Clone repositories
git clone https://github.com/your-org/repo1.git
git clone https://github.com/your-org/repo2.git

# Copy relevant files to data/raw/github/
# Example: copy only Python and docs
find repo1/ -name "*.py" -o -name "*.md" | xargs -I {} cp {} data/raw/github/
```

**File Types to Include:**
```
Code:
- .py (Python)
- .js, .ts (JavaScript/TypeScript)
- .java (Java)
- .cpp, .c, .h (C/C++)
- .go (Go)
- .rs (Rust)

Documentation:
- .md (Markdown)
- .txt (Text)
- README files
- CHANGELOG files
```

**Best Practices:**
- Focus on well-documented code
- Include inline comments
- Add README files for context
- Remove generated code
- Exclude binary files

### 2.7 Email (Outlook, Gmail)

**What to Export:**
- Business correspondence
- Project updates
- Technical discussions

**Gmail Export:**
1. Go to [takeout.google.com](https://takeout.google.com)
2. Select "Mail"
3. Choose MBOX format
4. Download and extract

**Outlook Export:**
1. File → Open & Export → Import/Export
2. Export to a file → Outlook Data File (.pst)
3. Select folders to export
4. Save and extract

**Processing Emails:**
```python
# Example: Extract email text
import mailbox

mbox = mailbox.mbox('exported_emails.mbox')
for message in mbox:
    subject = message['subject']
    body = message.get_payload()
    # Save to text files
```

**Best Practices:**
- Remove personal/sensitive information
- Filter by sender/recipient domain
- Focus on business emails
- Remove email signatures
- Clean HTML formatting

### 2.8 Confluence / Wiki

**What to Export:**
- Documentation pages
- Project wikis
- Knowledge base articles

**How to Export:**

**Confluence:**
1. Space Tools → Export Space
2. Choose "HTML" or "PDF"
3. Download and extract

**MediaWiki:**
```bash
# Export using dumpBackup.php
php maintenance/dumpBackup.php --current > wiki-export.xml
```

**Best Practices:**
- Convert to plain text or markdown
- Preserve document hierarchy
- Include page titles
- Remove wiki markup

### 2.9 Jira / Project Management

**What to Export:**
- Issue descriptions
- Comments
- Documentation fields
- Project summaries

**Jira Export:**
1. Filter issues
2. Export → CSV or JSON
3. Download and save

**Best Practices:**
- Focus on completed issues
- Include comments for context
- Remove system fields
- Organize by project

### 2.10 Local Documents

**What to Include:**
- Any relevant text documents
- PDFs, Word docs, presentations
- Text files, markdown files
- Reports and analyses

**Organization:**
```
local/
├── reports/
│   ├── q1_report.pdf
│   └── q2_report.docx
├── presentations/
│   └── product_pitch.pptx
├── documentation/
│   ├── user_guide.md
│   └── api_docs.txt
└── policies/
    └── security_policy.pdf
```

---

## Part 3: Data Organization

### Recommended Directory Structure

```
data/
├── raw/                    # Original, unprocessed data
│   ├── sharepoint/
│   │   ├── dept_finance/
│   │   ├── dept_engineering/
│   │   └── dept_marketing/
│   ├── google_drive/
│   │   ├── shared_docs/
│   │   └── team_folders/
│   ├── slack/
│   │   ├── general/
│   │   ├── engineering/
│   │   └── support/
│   ├── fireflies/
│   │   ├── 2024-Q1/
│   │   └── 2024-Q2/
│   ├── github/
│   │   ├── backend/
│   │   ├── frontend/
│   │   └── docs/
│   └── local/
│       ├── reports/
│       ├── policies/
│       └── misc/
└── processed/              # Auto-generated by scripts
    ├── processed_corpus.jsonl
    └── corpus_statistics.json
```

---

## Part 4: Data Preprocessing Tips

### Text Cleaning Recommendations

**Enable in config.py:**
```python
PREPROCESSING_CONFIG = {
    'remove_urls': True,           # Remove http:// links
    'remove_emails': True,         # Remove email addresses
    'remove_phone_numbers': False, # Keep for context
    'normalize_whitespace': True,  # Clean up spacing
    'remove_html': True,          # Remove HTML tags
    
    'min_doc_length': 50,         # Skip very short docs
    'max_doc_length': 50000,      # Skip extremely long docs
    
    'remove_duplicates': True,     # Important!
    'preserve_code_formatting': True,  # For code files
}
```

### Handling Different Document Types

**PDFs:**
- Ensure OCR quality is good
- Check for table extraction issues
- Verify text ordering

**Code Files:**
- Keep comments and docstrings
- Preserve indentation
- Include README context

**Spreadsheets:**
- Extract meaningful text, not just numbers
- Include headers for context
- Skip pure numerical tables

**Presentations:**
- Combine slide titles and content
- Keep speaker notes if available
- Preserve slide order

---

## Part 5: Data Quality Checks

### Before Processing

Run these checks on your raw data:

```bash
# Count files
find data/raw -type f | wc -l

# Check file sizes
du -sh data/raw/*

# List file types
find data/raw -type f | sed 's/.*\.//' | sort | uniq -c

# Sample random files
find data/raw -type f | shuf -n 5
```

### After Processing

Check processed corpus quality:

```python
import json

# Load corpus
with open('data/processed/processed_corpus.jsonl', 'r') as f:
    docs = [json.loads(line) for line in f]

# Statistics
print(f"Total documents: {len(docs)}")
print(f"Avg length: {sum(len(d['text'].split()) for d in docs) / len(docs):.0f} words")
print(f"Sources: {set(d['source'] for d in docs)}")

# Sample documents
import random
sample = random.choice(docs)
print(f"\nSample document ({sample['source']}):")
print(sample['text'][:500] + "...")
```

---

## Part 6: Common Issues and Solutions

### Issue: Too Few Documents

**Symptoms:**
- Corpus < 1000 documents
- Training completes too quickly
- Poor generation quality

**Solutions:**
1. Export more data sources
2. Split large documents into sections
3. Include more time periods
4. Add documentation and wikis

### Issue: Too Many Duplicates

**Symptoms:**
- Processing removes 50%+ documents
- Many identical files
- Version copies

**Solutions:**
1. Enable deduplication in config
2. Clean source directories first
3. Remove backup/draft files
4. Keep only final versions

### Issue: Poor Text Quality

**Symptoms:**
- Lots of garbled text
- Excessive special characters
- Broken formatting

**Solutions:**
1. Check PDF extraction quality
2. Adjust preprocessing settings
3. Manually clean problematic files
4. Use better OCR if needed

### Issue: Imbalanced Data

**Symptoms:**
- 90% from one source
- One document type dominates
- Single topic focus

**Solutions:**
1. Balance data sources
2. Limit over-represented categories
3. Add diverse content
4. Adjust sampling in config

---

## Part 7: Privacy and Compliance

### Data Privacy Checklist

Before including any data:

- [ ] Ensure you have authorization to use this data
- [ ] Remove personally identifiable information (PII)
- [ ] Check compliance requirements (GDPR, HIPAA, etc.)
- [ ] Exclude confidential/sensitive information
- [ ] Review with legal/compliance team
- [ ] Document data sources and permissions
- [ ] Implement access controls
- [ ] Plan for data retention/deletion

### PII Removal Examples

**Automatic PII Removal:**
```python
import re

def remove_pii(text):
    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                  '<EMAIL>', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '<PHONE>', text)
    
    # Remove SSN-like patterns
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '<SSN>', text)
    
    # Remove credit card numbers
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 
                  '<CARD>', text)
    
    return text
```

**Enable in config.py:**
```python
PREPROCESSING_CONFIG = {
    'remove_emails': True,
    'remove_phone_numbers': True,
    # Add custom PII filters in data_loader.py
}
```

---

## Part 8: Dataset Examples

### Example 1: Small Company (1K employees)

**Data Sources:**
- 500 SharePoint documents
- 300 Google Drive docs
- 200 Slack exports (1 year)
- 50 meeting transcripts
- 100 GitHub README files

**Total:** ~1,200 documents, ~2M tokens  
**Training Time:** 4-6 hours  
**Model Quality:** Good for internal use

### Example 2: Medium Enterprise (10K employees)

**Data Sources:**
- 5,000 SharePoint documents
- 3,000 Google Drive docs
- 2,000 Slack exports (2 years)
- 500 meeting transcripts
- 500 GitHub code + docs
- 500 Confluence pages

**Total:** ~11,500 documents, ~15M tokens  
**Training Time:** 12-18 hours  
**Model Quality:** Production-ready

### Example 3: Large Organization (100K+ employees)

**Data Sources:**
- 50,000+ SharePoint documents
- 30,000+ Google Drive docs
- 10,000+ Slack exports (3 years)
- 2,000+ meeting transcripts
- 5,000+ GitHub repositories
- 3,000+ Confluence pages

**Total:** 100,000+ documents, 100M+ tokens  
**Training Time:** 2-5 days  
**Model Quality:** Enterprise production

---

## Part 9: Quick Start Checklist

### Week 1: Data Collection
- [ ] Identify data sources
- [ ] Get necessary permissions
- [ ] Export initial batch (1000+ docs)
- [ ] Organize in folder structure
- [ ] Run initial quality check

### Week 2: Processing & Training
- [ ] Configure data sources in config.py
- [ ] Run data_loader.py
- [ ] Train tokenizer
- [ ] Start initial training
- [ ] Monitor and iterate

### Week 3: Evaluation & Refinement
- [ ] Test model quality
- [ ] Add more data if needed
- [ ] Fine-tune on specific domains
- [ ] Deploy for testing
- [ ] Collect feedback

---

## Part 10: Dataset Maintenance

### Regular Updates

**Monthly:**
- Add new documents
- Remove outdated content
- Retrain tokenizer if vocab changes
- Fine-tune model incrementally

**Quarterly:**
- Full data review
- Balance data sources
- Update preprocessing rules
- Complete model retraining

**Annually:**
- Comprehensive data audit
- Archive old data
- Update compliance checks
- Major model updates

---

## Summary: Quick Reference

```
┌────────────────────────────────────────────────────────┐
│  DATA PREPARATION CHECKLIST                            │
├────────────────────────────────────────────────────────┤
│  ✓ Collect 1000+ documents minimum                     │
│  ✓ Organize in data/raw/ structure                     │
│  ✓ Enable sources in config.py                         │
│  ✓ Remove PII and sensitive data                       │
│  ✓ Check data quality                                  │
│  ✓ Run data_loader.py                                  │
│  ✓ Verify processed corpus                             │
│  ✓ Ready for tokenizer training!                       │
└────────────────────────────────────────────────────────┘
```

Your enterprise dataset is now ready for training! 🎯
