# Test Infrastructure

This directory contains scripts for testing the TDS Quiz Solver pipeline end-to-end.

## Overview

The test infrastructure consists of:

1. **`run_test_server.py`** - Local test server that serves sample quizzes
2. **`test_pipeline.py`** - Pipeline tester that validates the /solve endpoint
3. **`test_demo_endpoint.py`** - Test against the official TDS demo endpoint
4. **`test_prompts.py`** - Interactive prompt testing utility
5. **`quick_test.py`** - Quick comparison of viva prompts
6. **`demo_test.py`** - Demo test utilities
7. **`run_demo_agent.py`** - Run the agent in demo mode

## Quick Start

### Option A: Test Against Official Demo Endpoint (Recommended)

The official TDS demo endpoint simulates the quiz process and is the recommended way to test your implementation:

```bash
python scripts/test_demo_endpoint.py
```

This will:
1. Load credentials from your `.env` file
2. POST to your `/solve` endpoint with the demo quiz URL
3. Verify you receive an immediate 200 response
4. Display detailed results

**Official Demo Quiz URL**: `https://tds-llm-analysis.s-anand.net/demo`

**Expected Payload**:
```json
{
  "email": "your email",
  "secret": "your secret",
  "url": "https://tds-llm-analysis.s-anand.net/demo"
}
```

### Option B: Test Against Local Test Server

For development and debugging, use the local test server:

#### 1. Start the Test Server

The test server provides a local quiz endpoint for testing without hitting real quiz servers:

```bash
python scripts/run_test_server.py --port 8888
```

This will start a server at `http://localhost:8888` with:
- Quiz page: `http://localhost:8888/quiz-834`
- Data file: `http://localhost:8888/data.csv`
- Submit endpoint: `http://localhost:8888/submit-quiz-834`

The sample quiz asks: "What is the sum of all values in the 'amount' column?"
- Correct answer: **650**

#### 2. Start the Main API Server

In a separate terminal, start the main API server:

```bash
uvicorn app.main:app --reload --port 8000
```

#### 3. Run the Pipeline Test

Test the complete end-to-end flow:

```bash
python scripts/test_pipeline.py --url http://localhost:8888/quiz-834
```

This will:
1. Check API server health
2. Test malformed request handling (should return 400)
3. POST to `/solve` endpoint with the quiz URL
4. Verify immediate 200 response
5. Display response time and body

#### Expected Output

```
==================================================================================
TDS Quiz Solver - Pipeline Test Suite
==================================================================================

[1/3] Health Check
------------------------------------------------------------------------------
✓ API server is healthy

[2/3] Malformed Request Test
------------------------------------------------------------------------------
✓ Malformed request correctly rejected with 400

[3/3] Solve Endpoint Test
------------------------------------------------------------------------------
✓ /solve returned 200 OK in 45.23ms (background processing started)

==================================================================================
Test Summary
==================================================================================
Quiz URL: http://localhost:8888/quiz-834
API Base URL: http://localhost:8000
Response Time: 45.23ms
Response Body: {
  "status": "ok",
  "accepted": true
}

✓ All tests passed!

Note: Background processing is asynchronous. Check the API server logs to verify:
  - Quiz was scraped successfully
  - Task was parsed correctly
  - Agent computed the answer
  - Answer was submitted
  - Relative URLs were resolved correctly
```

## Demo Endpoint Testing

### Testing Against Official Demo

The official demo endpoint is provided by TDS to validate your implementation:

```bash
# Basic test (uses .env credentials)
python scripts/test_demo_endpoint.py

# Test with explicit credentials
python scripts/test_demo_endpoint.py --email your@email.com --secret your-secret

# Test with custom API endpoint (for deployed servers)
python scripts/test_demo_endpoint.py --api https://your-server.com

# Verbose output for debugging
python scripts/test_demo_endpoint.py -v
```

### Expected Behavior

1. **Immediate 200 Response**: Your `/solve` endpoint should return HTTP 200 immediately to acknowledge receipt
2. **Background Processing**: The quiz solving happens asynchronously in the background
3. **Complete Workflow**:
   - Browser scrapes the quiz page
   - LLM parses task instructions and payload structure
   - Agent solves the quiz using available tools
   - Answer is submitted to the quiz's submission endpoint
   - System handles any chained quizzes

### Response Codes

- **200 OK**: Request accepted, quiz solving started in background ✓
- **400 Bad Request**: Malformed JSON payload (missing required fields)
- **403 Forbidden**: Invalid email/secret combination

### Success Output Example

```
==================================================================================
Testing Official TDS Demo Endpoint
==================================================================================
Demo Quiz URL: https://tds-llm-analysis.s-anand.net/demo
Your API Endpoint: http://localhost:8000/solve
Email: your@email.com
Timeout: 60.0s

Sending POST request to /solve endpoint...

✓ SUCCESS: /solve returned 200 OK in 45.23ms

Response body:
{
  "status": "ok",
  "accepted": true
}

Your endpoint is working correctly!
The quiz is being solved in the background.
```

## Test Server Details

### Sample Quiz Structure

The test server serves a quiz with the following structure:

**Question**: "Given the CSV file below, what is the sum of all values in the 'amount' column?"

**Resources**:
- `/data.csv` - CSV file with an 'amount' column containing: 100, 150, 75, 200, 125

**Submission Endpoint**: `/submit-quiz-834`

**Expected Payload Structure**:
```json
{
    "quiz_id": "quiz-834",
    "email": "{{email}}",
    "secret": "{{secret}}",
    "answer": "{{answer}}",
    "metadata": {
        "reasoning": "{{reasoning}}"
    }
}
```

This tests the payload template extraction feature - the quiz HTML contains a JSON structure that should be extracted and used for submission.

### API Endpoints

- `GET /` - Server info and endpoint list
- `GET /quiz-834` - Sample quiz HTML
- `GET /data.csv` - Sample CSV data
- `POST /submit-quiz-834` - Mock submission endpoint (accepts quiz answers)
- `GET /health` - Health check

## Pipeline Tester Options

### Basic Usage

```bash
# Test against official demo endpoint
python scripts/test_pipeline.py --demo

# Test against local test server
python scripts/test_pipeline.py --url http://localhost:8888/quiz-834

# Test against custom quiz URL
python scripts/test_pipeline.py --url <QUIZ_URL>
```

### Advanced Options

```bash
# Custom API endpoint
python scripts/test_pipeline.py --demo --api http://localhost:8000

# Provide credentials explicitly (otherwise uses .env)
python scripts/test_pipeline.py --demo --email test@example.com --secret my-secret

# Skip health check
python scripts/test_pipeline.py --demo --skip-health

# Skip malformed request test
python scripts/test_pipeline.py --demo --skip-malformed

# Verbose output
python scripts/test_pipeline.py --demo -v
```

## Monitoring Background Processing

After the `/solve` endpoint returns 200, the actual quiz solving happens in the background. To verify it worked:

1. **Check API Server Logs** (terminal running uvicorn):
   - Look for "Background task started"
   - "Scraping quiz page"
   - "Parsing task with LLM"
   - "Agent solving quiz"
   - "Submitting answer"
   - "Quiz solving completed"

2. **Check Test Server Logs** (if using local test server):
   - Look for "Received submission"
   - "✓ Correct answer received: 650" (if answer is correct)

## Integration Tests

The repository also includes integration tests in `tests/test_sample_pipeline.py`:

```bash
# Run all integration tests
pytest tests/test_sample_pipeline.py -v

# Run specific test
pytest tests/test_sample_pipeline.py::test_sample_quiz_scraping -v
```

These tests validate:
- Quiz HTML scraping
- Task parsing and payload template extraction
- Agent solving with correct answer
- Submission with payload template
- Relative URL resolution
- End-to-end pipeline flow

## Troubleshooting

### Connection Error

```
✗ Connection error - is the API server running at http://localhost:8000?
```

**Solution**: Start the API server with `uvicorn app.main:app --port 8000`

### Timeout

```
✗ Request timeout after 30.0s
```

**Solution**: 
- Check API server logs for errors
- Verify all dependencies are installed
- Check that .env file has required credentials

### 403 Forbidden

```
✗ /solve returned 403
```

**Solution**: Verify `EMAIL` and `SECRET` in your `.env` file match the server's expected credentials

### Background Task Not Running

If `/solve` returns 200 but nothing happens in the background:

1. Check API server logs for errors
2. Verify browser dependencies: `playwright install chromium`
3. Check that LLM API keys are set in `.env`

## Environment Variables

Required in `.env`:

```env
EMAIL=your-email@example.com
SECRET=your-secret-key
AIPIPE_API_KEY=your-aipipe-key
```

Optional (for testing):

```env
LOG_LEVEL=DEBUG  # More verbose logging
PLAYWRIGHT_HEADLESS=false  # See browser in action
```

## Pre-Submission Checklist

Before submitting your endpoint to TDS:

1. **Test Locally**:
   ```bash
   python scripts/test_demo_endpoint.py
   ```

2. **Verify Background Processing**:
   - Check API server logs for successful quiz solving
   - Ensure answer submission completes without errors

3. **Deploy to Public Server**:
   - Deploy your API to a publicly accessible HTTPS endpoint
   - Test the deployed endpoint:
     ```bash
     python scripts/test_demo_endpoint.py --api https://your-server.com
     ```

4. **Final Validation**:
   - Verify your endpoint returns 200 immediately
   - Confirm background processing completes successfully
   - Test with the official demo endpoint multiple times

5. **Submit**:
   - Submit your public endpoint URL via the Google Form
   - Include your email and secret that your endpoint will use

## See Also

- Main API documentation: `../README.md`
- Integration tests: `../tests/test_sample_pipeline.py`
- Orchestrator tests: `../tests/test_orchestrator.py`
