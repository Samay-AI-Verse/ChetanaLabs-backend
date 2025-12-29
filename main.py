import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from authlib.integrations.starlette_client import OAuth, OAuthError
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware
from datetime import datetime
from bson import ObjectId # Import ObjectId for explicit conversion
import json
from groq import Groq  # IMPORT GROQ
from pydantic import BaseModel
import requests  # For Vapi API calls
# 1. Load Config
load_dotenv()

# 2. App Setup
app = FastAPI()

# Add Session Middleware (Required for OAuth)
# WARNING: Ensure SECRET_KEY is set in your .env file
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

# 3. Path Setup (Connecting to your sibling Frontend folder)
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "Frontend"

if not FRONTEND_DIR.exists():
    raise RuntimeError(f"Frontend directory not found at {FRONTEND_DIR}")

# 4. Mount Static Files
# This makes style.css, script.js, and images available to the browser
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/image", StaticFiles(directory=str(FRONTEND_DIR / "image")), name="images")
# 5. Database Setup (MongoDB Atlas)
@app.on_event("startup")
async def startup_db_client():
    mongo_url = os.getenv("MONGODB_URL")
    if not mongo_url:
        print("❌ ERROR: MONGODB_URL is missing in .env")
        return
    app.mongodb_client = AsyncIOMotorClient(mongo_url)
    app.mongodb = app.mongodb_client[os.getenv("DB_NAME")]
    print("✅ Connected to MongoDB Atlas (Online)")

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()

# 6. Google OAuth Setup
oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# --- ROUTES ---

# Route 1: The Entry Point (Login Form)
@app.get("/")
async def read_root():
    return FileResponse(FRONTEND_DIR / "loginform.html")

# Route 2: Start Google Login
@app.get('/login/google')
async def login_google(request: Request):
    redirect_uri = request.url_for('auth_google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

# Route 3: Google Callback (The Magic Happens Here)
@app.get('/auth/google/callback')
async def auth_google_callback(request: Request):
    try:
        # 1. Get Token & User Info from Google
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        
        if not user_info:
            raise HTTPException(status_code=400, detail="Failed to get user info")

        # 2. Save User to Online MongoDB Atlas
        users_collection = app.mongodb["users"]
        user_data = {
            "google_id": user_info.get("sub"),
            "email": user_info.get("email"),
            "name": user_info.get("name"),
            "picture": user_info.get("picture"),
            "last_login": datetime.utcnow().isoformat()
        }
        
        # Update if exists, Insert if new (Upsert)
        await users_collection.update_one(
            {"email": user_data["email"]},
            {"$set": user_data},
            upsert=True
        )

        # 3. Store user in session (cookie) so they stay logged in
        request.session['user'] = user_data

        # 4. Redirect to the Dashboard
        return RedirectResponse(url='/dashboard')

    except OAuthError as e:
        return {"error": f"OAuth Error: {e.error}"}

# Route 4: The Dashboard (Index.html)
@app.get("/dashboard")
async def dashboard(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/') # Kick back to login if not signed in
        
    return FileResponse(FRONTEND_DIR / "index.html")

# Helper to serve other files if needed (like images referenced in HTML)
@app.get("/{filename}")
async def serve_root_files(filename: str):
    file_path = FRONTEND_DIR / filename
    if file_path.is_file():
        return FileResponse(file_path)
    return {"error": "File not found"}


# --- USER PROFILE ROUTE ---
@app.get("/api/me")
async def get_current_user(request: Request):
    user = request.session.get('user')
    if not user:
        return {"error": "Not logged in"}
    return user


@app.post("/api/campaigns/create")
async def create_new_campaign_api(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    try:
        data = await request.json()
        campaign_name = data.get("name", "Untitled Campaign")
        # 1. Capture the type (default to 'audio' if not sent)
        campaign_type = data.get("type", "audio") 
    except Exception:
        campaign_name = "Untitled Campaign"
        campaign_type = "audio"

    new_campaign = {
        "user_id": user["google_id"],
        "name": campaign_name,
        "type": campaign_type, # 2. Save type to DB
        "created_at": datetime.utcnow().isoformat(),
        "status": "In Design",
        "candidate_count": 0,
        "config": {
            "mode": "Technical Round",
            "duration": "12 Mins",
            "script": "Initial script prompt...",
            "voice": "Sarah",
        }
    }

    campaigns_collection = app.mongodb["campaigns"]
    result = await campaigns_collection.insert_one(new_campaign)
    
    new_campaign["_id"] = str(new_campaign["_id"]) 

    return {
        "message": "Campaign created successfully",
        "id": str(result.inserted_id),
        "campaign": new_campaign
    }
# --- CAMPAIGN RETRIEVAL ROUTE ---
@app.get("/api/campaigns")
async def get_user_campaigns(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    campaigns_collection = app.mongodb["campaigns"]
    
    # Fetch all campaigns belonging to the logged-in user
    campaigns_cursor = campaigns_collection.find({"user_id": user["google_id"]}).sort("created_at", -1)
    
    campaigns_list = []
    async for doc in campaigns_cursor:
        # Crucial for GET route: Convert MongoDB ObjectId to string for JSON serialization
        doc["id"] = str(doc.pop("_id"))
        campaigns_list.append(doc)
        
    return {"campaigns": campaigns_list}



# --- Add these imports at the top of main.py ---
from fastapi import UploadFile, File, Form
import pandas as pd
import io
from pypdf import PdfReader

# --- IN main.py ---
# Replace your existing '/api/parse-candidates' with this smarter version

@app.post("/api/parse-candidates")
async def parse_candidates(file: UploadFile = File(...)):
    contents = await file.read()
    
    try:
        # 1. Load Data into Pandas
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Unsupported file format"}
        
        # 2. Smart Column Matcher
        # We look for these keywords in the Excel headers
        def get_column_by_keyword(df_columns, keywords):
            # Normalization helper
            normalize = lambda x: str(x).lower().replace("_", "").replace(" ", "").strip()
            
            for col in df_columns:
                col_norm = normalize(col)
                for kw in keywords:
                    if kw in col_norm:
                        return col
            return None

        # Define variations of keywords
        name_col = get_column_by_keyword(df.columns, ["name", "candidate", "fullname", "student"])
        email_col = get_column_by_keyword(df.columns, ["email", "mail", "gmail", "e-mail"])
        phone_col = get_column_by_keyword(df.columns, ["phone", "mobile", "contact", "cell", "number", "tel"])

        if not name_col and not email_col and not phone_col:
            return {"error": "Could not automatically identify Name, Email, or Phone columns."}

        # 3. Extraction & Cleaning Loop
        candidates = []
        
        # Convert NaN to None for easier handling
        df = df.where(pd.notnull(df), None)

        for _, row in df.iterrows():
            # Get raw values
            name_val = row[name_col] if name_col else "Unknown Candidate"
            email_val = row[email_col] if email_col else ""
            phone_val = row[phone_col] if phone_col else None

            # --- CRITICAL: SKIP IF PHONE IS MISSING ---
            # The user requested: "if empty value like mobile then don't take this"
            if not phone_val or str(phone_val).strip() == "":
                continue 

            # Clean Phone Number (Remove decimals like 9999.0 generated by Excel)
            phone_str = str(phone_val).split('.')[0].strip()

            candidates.append({
                "name": str(name_val).strip(),
                "email": str(email_val).strip(),
                "phone": phone_str
            })
            
        return {"candidates": candidates, "count": len(candidates)}
        
    except Exception as e:
        print(f"Error parsing file: {e}")
        return {"error": "Failed to parse file. Please check format."}

@app.post("/api/generate-questions")
async def generate_questions_api(
    job_role: str = Form(""),       # NEW FIELD
    context_text: str = Form(""),
    file: UploadFile = File(None)
):
    # 1. Parse PDF if exists
    pdf_text = ""
    if file:
        try:
            reader = PdfReader(io.BytesIO(await file.read()))
            for page in reader.pages:
                pdf_text += page.extract_text()
        except:
            pass

    # 2. Smart Prompt Construction (Mock AI)
    # In a real app, you would send this 'full_prompt' to OpenAI/Gemini
    
    questions = []
    
    # Dynamic header based on role
    role_title = job_role if job_role else "Candidate"
    questions.append(f"--- Interview Script for: {role_title} ---")
    questions.append("Objective: Evaluate technical skills and cultural fit.\n")

    # Generate context-aware questions
    if "python" in context_text.lower() or "python" in job_role.lower():
        questions.append("1. [Technical] Explain the difference between list and tuple in Python.")
        questions.append("2. [Scenario] How do you handle memory management in large datasets?")
    elif "sales" in job_role.lower():
        questions.append("1. [Behavioral] Describe a time you turned a 'No' into a 'Yes'.")
        questions.append("2. [Strategy] How do you prioritize your lead pipeline?")
    else:
        # Generic fallback
        questions.append(f"1. Can you walk us through your experience as a {role_title}?")
        questions.append("2. What is the most challenging project you've worked on recently?")

    questions.append("3. [Culture] How do you handle disagreements with team members?")
    questions.append("4. Do you have any questions for us about the role?")

    return {"questions": "\n".join(questions)}


# --- UPDATE THIS FUNCTION IN main.py ---
@app.post("/api/candidates/save")
async def save_candidates(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    payload = await request.json()
    campaign_id = payload.get("campaign_id")
    candidates = payload.get("candidates", [])
    
    if not campaign_id:
        raise HTTPException(status_code=400, detail="campaign_id is required")

    coll = app.mongodb["candidates"]
    
    # 1. Get existing phone numbers for this campaign to avoid duplicates
    existing_cursor = coll.find(
        {"campaign_id": campaign_id, "user_id": user["google_id"]},
        {"phone": 1}
    )
    existing_phones = set()
    async for doc in existing_cursor:
        if "phone" in doc:
            existing_phones.add(doc["phone"])

    # 2. Filter out duplicates
    new_candidates = []
    skipped_count = 0

    for c in candidates:
        clean_phone = str(c.get("phone", "")).strip()
        
        # Validation: Must have phone and NOT be in existing list
        if clean_phone and clean_phone not in existing_phones:
            c["campaign_id"] = campaign_id
            c["user_id"] = user["google_id"]
            c["created_at"] = datetime.utcnow().isoformat()
            c["status"] = "Pending"
            c["phone"] = clean_phone # Ensure clean phone is saved
            new_candidates.append(c)
            existing_phones.add(clean_phone) # Add to set to prevent dups within the same batch
        else:
            skipped_count += 1
    
    insert_count = 0
    if new_candidates:
        try:
            result = await coll.insert_many(new_candidates)
            insert_count = len(result.inserted_ids)
        except Exception as e:
            print(f"Database Insert Error: {e}")
            raise HTTPException(status_code=500, detail="Failed to save candidates")
            
    return {
        "message": "Sync complete",
        "added": insert_count,
        "skipped": skipped_count
    }

# --- ADD THIS NEW ENDPOINT TO main.py ---
@app.delete("/api/candidates/{candidate_id}")
async def delete_single_candidate(candidate_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)

    try:
        from bson import ObjectId
        res = await app.mongodb["candidates"].delete_one({
            "_id": ObjectId(candidate_id),
            "user_id": user["google_id"]
        })
        
        if res.deleted_count == 1:
            return {"status": "success"}
        raise HTTPException(status_code=404, detail="Candidate not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# --- GROQ (LLAMA 3) SETUP ---
# Initialize Groq Client
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

# --- VAPI SETUP ---
VAPI_API_KEY = os.getenv("VAPI_API_KEY")
VAPI_PHONE_NUMBER_ID = os.getenv("VAPI_PHONE_NUMBER_ID")
VAPI_BASE_URL = "https://api.vapi.ai"

# Helper: Make Vapi API Requests
def vapi_request(method: str, endpoint: str, data: dict = None):
    """Helper function to make requests to Vapi API"""
    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }
    url = f"{VAPI_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Vapi API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Vapi API Error: {str(e)}")

class BlueprintRequest(BaseModel):
    company_name: str
    job_role: str
    description: str
    candidate_count: int
    agent_persona: str
    strictness: str
    interview_mode: str
    duration: int

# --- FIND THIS FUNCTION IN main.py AND REPLACE IT ---

@app.post("/api/generate-blueprint")
async def generate_blueprint_api(request: Request):
    try:
        data = await request.json()
        
        # --- 1. Parse & Sanitize Inputs ---
        raw_duration = data.get('duration', 15)
        if isinstance(raw_duration, str):
            try:
                duration = int(raw_duration.split()[0])
            except:
                duration = 15 
        else:
            duration = int(raw_duration)

        # Calculate question count (approx 2.5 mins per question)
        target_q_count = max(2, int((duration - 3) / 2.5)) 
        
        # Parse fields
        mode = data.get('interview_mode', 'Technical')
        job_role = data.get('job_role', 'Candidate')
        job_domain = data.get('job_domain', '')
        tech_stack = data.get('tech_stack', '')
        hr_focus = data.get('hr_focus', '')

        # --- 2. DEFINE "REAL HUMAN" PERSONA ---
        # This logic changes HOW the AI speaks
        
        persona_instruction = ""
        
        if "HR" in mode:
            persona_instruction = """
            PERSONA IDENTITY: You are a warm, empathetic, and professional Senior HR Manager.
            - TONE: Conversational, encouraging, and natural. Use fillers like "That's interesting" or "I see."
            - BEHAVIOR: Do not simply ask list questions. If a candidate gives a short answer, probe deeper using the STAR method (Situation, Task, Action, Result).
            - PRIORITY: Assess cultural fit, communication skills, and career motivations.
            """
        elif "Technical" in mode:
            persona_instruction = """
            PERSONA IDENTITY: You are a Senior Technical Lead / Engineering Manager.
            - TONE: Professional, direct, analytical, and sharp.
            - BEHAVIOR: Focus on 'How' and 'Why'. If they mention a technology, ask why they chose it over alternatives. Test their depth.
            - PRIORITY: Verify actual hands-on expertise and problem-solving logic.
            """
        else: # Mixed
            persona_instruction = """
            PERSONA IDENTITY: You are a Hiring Manager looking for a well-rounded team member.
            - TONE: Balanced and professional.
            - BEHAVIOR: Switch naturally between discussing technical projects and team dynamics.
            - PRIORITY: Assess both technical competence and team fit.
            """

        # --- 3. BUILD CONTEXT BLOCKS ---
        
        # A. Technical Context
        tech_context_block = ""
        if job_domain or tech_stack:
            tech_context_block = f"""
            [TECHNICAL DEPTH REQUIREMENTS]
            - Domain: {job_domain}
            - Core Stack: {tech_stack}
            
            INSTRUCTIONS:
            1. Ask specific, scenario-based questions using {tech_stack}.
            2. Avoid generic definitions (e.g., "What is React?"). Instead ask: "How have you handled state management scaling in React?"
            """
            
            # Special Logic for Data Science (Mathematics)
            if job_domain == "DataScience":
                tech_context_block += """
                3. REQUIRED: Include 1-2 questions testing Mathematical foundations (Statistics, Probability, or Linear Algebra) relevant to their ML models.
                """
            
            # Special Logic for Backend/Fullstack (System Design)
            if job_domain in ["Backend", "FullStack", "DevOps"]:
                tech_context_block += """
                3. REQUIRED: Include 1 question on System Design, Architecture, or Database Scaling.
                """

        # B. HR Context
        hr_context_block = ""
        if hr_focus:
            hr_context_block = f"""
            [HR EVALUATION CRITERIA]
            - Focus Areas: {hr_focus}
            
            INSTRUCTIONS:
            1. For each focus area, ask a behavioral question.
            2. Example for 'Culture': "Describe a time you disagreed with a team member. How did you resolve it?"
            """

        # --- 4. CONSTRUCT FINAL SYSTEM PROMPT ---
        system_instruction = f"""
        {persona_instruction}
        
        JOB ROLE: {job_role} at {data.get('company_name')}.
        JOB CONTEXT: {data.get('description')}
        TOTAL DURATION: {duration} Minutes.
        
        {tech_context_block}
        
        {hr_context_block}
        
        TASK:
        Generate a structured JSON Interview Blueprint.
        The system_prompt within the JSON must force the AI Agent to adopt the Persona defined above.
        
        RESPONSE FORMAT (JSON ONLY):
        {{
            "system_prompt": "You are {data.get('agent_persona')}, a {job_role} recruiter at {data.get('company_name')}. {persona_instruction.replace(chr(10), ' ')} Your goal is to screen for: {tech_stack} {hr_focus}.",
            "estimated_hours": "{round((data.get('candidate_count') * duration)/60, 1)}",
            "phases": [
                {{
                    "name": "Introduction & Verification",
                    "time_limit": "2 mins",
                    "questions": [ 
                        {{ "speaker": "AI", "text": "Hi, this is {data.get('agent_persona')} from {data.get('company_name')}. Thanks for taking the time. I'm reviewing your application for the {job_role} position. Is now a good time to chat?" }} 
                    ]
                }},
                {{
                    "name": "Core Interview",
                    "time_limit": "{duration - 4} mins",
                    "questions": [
                         {{ "speaker": "AI", "text": "(Generate specific Question 1 based on Technical/HR context)" }},
                         {{ "speaker": "AI", "text": "(Generate specific Question 2...)" }},
                         {{ "speaker": "AI", "text": "(Generate specific Question 3...)" }}
                    ]
                }},
                {{
                    "name": "Closing",
                    "time_limit": "1 min",
                    "questions": [
                        {{ "speaker": "AI", "text": "Thank you for sharing that. I have all I need for now. Our team will be in touch shortly regarding the next steps. Have a great day!" }}
                    ]
                }}
            ]
        }}
        """

        # --- 5. Call Groq API ---
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a JSON-only API. Output ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": system_instruction,
                }
            ],
            model="llama-3.3-70b-versatile", 
            temperature=0.7, # Slightly higher for more creative/human-like questions
            response_format={"type": "json_object"}, 
        )

        response_content = chat_completion.choices[0].message.content
        blueprint = json.loads(response_content)
        return blueprint

    except Exception as e:
        print(f"Groq API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/campaigns/{campaign_id}/save-final")
async def save_final_campaign(campaign_id: str, request: Request):
    """Saves the final Blueprint + Config to MongoDB"""
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401, detail="Not logged in")
    
    try:
        data = await request.json()
        
        # Update the campaign in MongoDB
        db = app.mongodb["campaigns"]
        
        # Ensure we use ObjectId for the query
        from bson import ObjectId
        
        await db.update_one(
            {"_id": ObjectId(campaign_id)}, 
            {"$set": {
                "status": "Ready",
                "blueprint": data.get('blueprint'),
                "config": data.get('config'),
                "updated_at": datetime.utcnow().isoformat()
            }}
        )
        return {"status": "success", "message": "Campaign saved successfully"}
        
    except Exception as e:
        print(f"Save Error: {e}")
        raise HTTPException(status_code=500, detail="Database save failed")


api_key = os.getenv("GROQ_API_KEY")
if api_key:
    print(f"✅ Key found: {api_key[:5]}...") # Prints first 5 chars only
else:
    print("❌ API Key NOT found! Check .env file.")



# --- ADD THIS TO main.py ---

from bson import ObjectId

# 1. Get Specific Campaign Details (Config + Blueprint)
@app.get("/api/campaigns/{campaign_id}")
async def get_campaign_details(campaign_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)

    db = app.mongodb["campaigns"]
    campaign = await db.find_one({"_id": ObjectId(campaign_id)})
    
    if campaign:
        campaign["id"] = str(campaign.pop("_id"))
        return campaign
    raise HTTPException(status_code=404, detail="Campaign not found")

# 2. Get Candidates for a Campaign
@app.get("/api/campaigns/{campaign_id}/candidates")
async def get_campaign_candidates(campaign_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)

    db = app.mongodb["candidates"]
    cursor = db.find({"campaign_id": campaign_id})
    
    candidates = []
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        candidates.append(doc)
        
    return {"candidates": candidates}



# --- ADD THIS TO main.py ---

@app.delete("/api/campaigns/{campaign_id}")
async def delete_campaign(campaign_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)

    try:
        from bson import ObjectId
        
        # 1. Delete the Campaign
        camp_res = await app.mongodb["campaigns"].delete_one({
            "_id": ObjectId(campaign_id),
            "user_id": user["google_id"] # Security check
        })

        if camp_res.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Campaign not found or access denied")

        # 2. Cleanup: Delete associated candidates
        await app.mongodb["candidates"].delete_many({"campaign_id": campaign_id})

        return {"status": "success", "message": "Campaign deleted"}

    except Exception as e:
        print(f"Delete Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete campaign")




# --- IN main.py ---
# Add this endpoint to fetch Dashboard Stats

@app.get("/api/dashboard-stats")
async def get_dashboard_stats(request: Request):
    user = request.session.get('user')
    if not user:
        return {"error": "Not logged in"}

    # Connect to collections
    candidates_coll = app.mongodb["candidates"]
    campaigns_coll = app.mongodb["campaigns"]
    user_id = user["google_id"]

    # 1. Count Total Candidates
    total_candidates = await candidates_coll.count_documents({"user_id": user_id})

    # 2. Count "Active" (Pending or In Progress)
    active_count = await candidates_coll.count_documents({
        "user_id": user_id, 
        "status": {"$in": ["Pending", "In Progress", "Scheduled"]}
    })

    # 3. Count "Completed" (Selected or Rejected)
    completed_count = await candidates_coll.count_documents({
        "user_id": user_id,
        "status": {"$in": ["Selected", "Rejected"]}
    })

    # 4. Count Total Campaigns
    total_campaigns = await campaigns_coll.count_documents({"user_id": user_id})

    return {
        "total_candidates": total_candidates,
        "active_candidates": active_count,
        "interviews_done": completed_count,
        "total_campaigns": total_campaigns
    }



# --- ADD THESE ENDPOINTS TO THE BOTTOM OF main.py ---

import random

# 1. DASHBOARD STATS (Counts real data from DB)
@app.get("/api/dashboard-stats")
async def get_dashboard_stats(request: Request):
    user = request.session.get('user')
    if not user: return {"error": "Not logged in"}

    user_id = user["google_id"]
    candidates_coll = app.mongodb["candidates"]
    campaigns_coll = app.mongodb["campaigns"]

    # Count actual documents
    total_candidates = await candidates_coll.count_documents({"user_id": user_id})
    
    # Count "Active" (Pending/Scheduled)
    active_count = await candidates_coll.count_documents({
        "user_id": user_id, 
        "status": {"$in": ["Pending", "In Progress", "Scheduled"]}
    })

    # Count "Completed" (Selected/Rejected)
    completed_count = await candidates_coll.count_documents({
        "user_id": user_id,
        "status": {"$in": ["Selected", "Rejected"]}
    })

    return {
        "total_candidates": total_candidates,
        "active_candidates": active_count,
        "interviews_done": completed_count
    }

# 2. LIVE ACTIVITY (Simulates "Active" status for the Dashboard table)
@app.get("/api/dashboard/live-activity")
async def get_live_activity(request: Request):
    user = request.session.get('user')
    if not user: return {"activity": []}

    # Fetch 5 most recent candidates
    cursor = app.mongodb["candidates"].find(
        {"user_id": user["google_id"]}
    ).sort("created_at", -1).limit(5)
    
    candidates = []
    
    # Simulation Logic: Give them "Live" statuses for the UI effect
    demo_statuses = ["Active Session", "Analyzed", "Dialing...", "Scheduled"]
    
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        
        # If status is default (Pending), assign a random "Live" status
        if doc.get("status", "Pending") == "Pending":
            doc["display_status"] = random.choice(demo_statuses)
            doc["duration"] = "04m 12s" if "Active" in doc["display_status"] else "--"
        else:
            doc["display_status"] = doc.get("status")
            doc["duration"] = doc.get("duration", "--")
            
        candidates.append(doc)
        
    return {"activity": candidates}

# 3. ALL CANDIDATES LIST (For the Candidates Tab)
@app.get("/api/candidates/all")
async def get_all_candidates_list(request: Request):
    user = request.session.get('user')
    if not user: return {"candidates": []}

    cursor = app.mongodb["candidates"].find({"user_id": user["google_id"]}).sort("created_at", -1)
    candidates = []
    
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        # Ensure a default status exists
        if "status" not in doc: doc["status"] = "Not Contacted"
        candidates.append(doc)
        
    return {"candidates": candidates}



# ============================================
# VAPI INTEGRATION ENDPOINTS
# ============================================

# 1. GET AVAILABLE VAPI ASSISTANTS
@app.get("/api/vapi/assistants")
async def get_vapi_assistants(request: Request):
    """Fetch all available Vapi assistants from the user's Vapi account"""
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401, detail="Not logged in")
    
    if not VAPI_API_KEY:
        # Return demo data if no API key configured
        return {
            "assistants": [
                {"id": "demo_gpt4", "name": "GPT-4 Interviewer", "model": "gpt-4", "provider": "openai"},
                {"id": "demo_claude", "name": "Claude Assistant", "model": "claude-3", "provider": "anthropic"},
                {"id": "demo_llama", "name": "Llama 3 Agent", "model": "llama-3-70b", "provider": "groq"}
            ]
        }
    
    try:
        data = vapi_request("GET", "/assistant")
        return {"assistants": data}
    except Exception as e:
        print(f"Error fetching Vapi assistants: {e}")
        return {"assistants": []}


# 2. GET AVAILABLE VOICE MODELS
@app.get("/api/vapi/voices")
async def get_vapi_voices(request: Request):
    """Fetch all available voice models from Vapi"""
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401, detail="Not logged in")
    
    # Return curated list of voices (Vapi supports multiple providers)
    return {
        "voices": [
            # 11Labs Voices
            {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel (11Labs)", "provider": "11labs", "language": "en-US", "gender": "female"},
            {"id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi (11Labs)", "provider": "11labs", "language": "en-US", "gender": "female"},
            {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella (11Labs)", "provider": "11labs", "language": "en-US", "gender": "female"},
            {"id": "ErXwobaYiN019PkySvjV", "name": "Antoni (11Labs)", "provider": "11labs", "language": "en-US", "gender": "male"},
            {"id": "VR6AewLTigWG4xSOukaG", "name": "Arnold (11Labs)", "provider": "11labs", "language": "en-US", "gender": "male"},
            
            # PlayHT Voices
            {"id": "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json", 
             "name": "Jennifer (PlayHT)", "provider": "playht", "language": "en-US", "gender": "female"},
            {"id": "larry", "name": "Larry (PlayHT)", "provider": "playht", "language": "en-US", "gender": "male"},
            
            # Azure Voices
            {"id": "en-US-JennyNeural", "name": "Jenny (Azure)", "provider": "azure", "language": "en-US", "gender": "female"},
            {"id": "en-US-GuyNeural", "name": "Guy (Azure)", "provider": "azure", "language": "en-US", "gender": "male"},
        ]
    }


# 3. CREATE VAPI ASSISTANT FROM CAMPAIGN
@app.post("/api/vapi/create-assistant")
async def create_vapi_assistant(request: Request):
    """Create a Vapi assistant based on campaign configuration"""
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401, detail="Not logged in")
    
    data = await request.json()
    campaign_id = data.get("campaign_id")
    
    if not campaign_id:
        raise HTTPException(status_code=400, detail="campaign_id is required")
    
    # Fetch campaign from DB
    campaign = await app.mongodb["campaigns"].find_one({"_id": ObjectId(campaign_id)})
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    # Get blueprint and config
    blueprint = campaign.get("blueprint", {})
    config = campaign.get("config", {})
    
    # Get selected voice and model from request
    voice_id = data.get("voice_id", "21m00Tcm4TlvDq8ikWAM")  # Default: Rachel
    voice_provider = data.get("voice_provider", "11labs")
    model_provider = data.get("model_provider", "openai")
    model_name = data.get("model_name", "gpt-4")
    
    # Build system prompt from blueprint or config
    system_prompt = blueprint.get("system_prompt", f"""
    You are a professional HR interviewer conducting a {config.get('mode', 'Technical')} interview 
    for the position of {config.get('job_role', 'Candidate')} at {config.get('company_name', 'our company')}.
    
    Be conversational, professional, and engaging. Ask follow-up questions based on candidate responses.
    """)
    
    # Prepare first message
    first_message = f"Hi, thanks for taking the time to speak with me today. I'm calling regarding your application for the {config.get('job_role', 'position')} role at {config.get('company_name', 'our company')}. Is now a good time to chat?"
    
    if not VAPI_API_KEY:
        # Demo mode: Just return a fake assistant ID
        fake_assistant_id = f"demo_assistant_{campaign_id[:8]}"
        await app.mongodb["campaigns"].update_one(
            {"_id": ObjectId(campaign_id)},
            {"$set": {
                "vapi_assistant_id": fake_assistant_id,
                "vapi_voice_id": voice_id,
                "vapi_config": {
                    "voice_provider": voice_provider,
                    "model_provider": model_provider,
                    "model_name": model_name
                }
            }}
        )
        return {
            "status": "success",
            "assistant_id": fake_assistant_id,
            "message": "Demo mode: Assistant ID saved (Vapi API key not configured)"
        }
    
    # Real Vapi API call
    try:
        assistant_data = {
            "name": f"{campaign.get('name', 'Interview')} - AI Agent",
            "model": {
                "provider": model_provider,
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ],
                "temperature": 0.7
            },
            "voice": {
                "provider": voice_provider,
                "voiceId": voice_id
            },
            "firstMessage": first_message,
            "endCallFunctionEnabled": True,
            "recordingEnabled": True,
            "analysisPlan": {
                "summaryPrompt": "Provide a structured summary of the candidate's technical skills, communication ability, and overall suitability for the role.",
                "structuredDataPrompt": "Extract: technical_skills (array), soft_skills (array), years_experience (number), recommendation (hire/maybe/pass)"
            },
            "serverUrl": f"{request.base_url}webhook/vapi/assistant-request",
            "endCallMessage": "Thank you for your time. Our team will be in touch with you shortly regarding the next steps. Have a great day!"
        }
        
        vapi_response = vapi_request("POST", "/assistant", assistant_data)
        assistant_id = vapi_response.get("id")
        
        # Save assistant ID to campaign
        await app.mongodb["campaigns"].update_one(
            {"_id": ObjectId(campaign_id)},
            {"$set": {
                "vapi_assistant_id": assistant_id,
                "vapi_voice_id": voice_id,
                "vapi_config": {
                    "voice_provider": voice_provider,
                    "model_provider": model_provider,
                    "model_name": model_name
                },
                "updated_at": datetime.utcnow().isoformat()
            }}
        )
        
        return {
            "status": "success",
            "assistant_id": assistant_id,
            "message": "Vapi assistant created successfully"
        }
        
    except Exception as e:
        print(f"Error creating Vapi assistant: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create assistant: {str(e)}")


# 4. START CAMPAIGN CALLS
@app.post("/api/vapi/start-campaign")
async def start_campaign_calls(request: Request):
    """Initiate outbound calls to all pending candidates in a campaign"""
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401, detail="Not logged in")
    
    data = await request.json()
    campaign_id = data.get("campaign_id")
    
    if not campaign_id:
        raise HTTPException(status_code=400, detail="campaign_id is required")
    
    # Fetch campaign
    campaign = await app.mongodb["campaigns"].find_one({"_id": ObjectId(campaign_id)})
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    assistant_id = campaign.get("vapi_assistant_id")
    if not assistant_id:
        raise HTTPException(status_code=400, detail="No Vapi assistant configured. Create one first.")
    
    # Fetch pending candidates
    candidates_cursor = app.mongodb["candidates"].find({
        "campaign_id": campaign_id,
        "status": {"$in": ["Pending", "Not Contacted"]}
    })
    
    results = []
    call_count = 0
    
    async for candidate in candidates_cursor:
        phone = candidate.get("phone", "")
        
        # Basic phone validation
        if not phone or len(phone) < 10:
            results.append({
                "candidate": candidate.get("name"),
                "status": "skipped",
                "reason": "Invalid phone number"
            })
            continue
        
        # Format phone to E.164 if needed (assuming US numbers for demo)
        formatted_phone = phone if phone.startswith("+") else f"+1{phone.replace('-', '').replace(' ', '')}"
        
        if not VAPI_API_KEY or not VAPI_PHONE_NUMBER_ID:
            # Demo mode: Just update status
            await app.mongodb["candidates"].update_one(
                {"_id": candidate["_id"]},
                {"$set": {
                    "status": "Scheduled (Demo)",
                    "vapi_call_id": f"demo_call_{candidate['_id']}",
                    "call_started_at": datetime.utcnow().isoformat()
                }}
            )
            results.append({
                "candidate": candidate.get("name"),
                "status": "demo_scheduled",
                "phone": formatted_phone
            })
            call_count += 1
            continue
        
        # Real Vapi API call
        try:
            call_data = {
                "assistantId": assistant_id,
                "customer": {
                    "number": formatted_phone,
                    "name": candidate.get("name", "Candidate")
                },
                "phoneNumberId": VAPI_PHONE_NUMBER_ID
            }
            
            vapi_response = vapi_request("POST", "/call/phone", call_data)
            call_id = vapi_response.get("id")
            
            # Update candidate with call info
            await app.mongodb["candidates"].update_one(
                {"_id": candidate["_id"]},
                {"$set": {
                    "status": "In Progress",
                    "vapi_call_id": call_id,
                    "call_started_at": datetime.utcnow().isoformat()
                }}
            )
            
            results.append({
                "candidate": candidate.get("name"),
                "status": "call_initiated",
                "call_id": call_id
            })
            call_count += 1
            
        except Exception as e:
            print(f"Error initiating call for {candidate.get('name')}: {e}")
            results.append({
                "candidate": candidate.get("name"),
                "status": "failed",
                "error": str(e)
            })
    
    # Update campaign status
    await app.mongodb["campaigns"].update_one(
        {"_id": ObjectId(campaign_id)},
        {"$set": {
            "status": "In Progress",
            "calls_initiated": call_count,
            "last_call_time": datetime.utcnow().isoformat()
        }}
    )
    
    return {
        "status": "success",
        "total_calls": call_count,
        "results": results
    }


# 5. WEBHOOK: CALL ENDED
@app.post("/webhook/vapi/call-ended")
async def vapi_call_ended_webhook(request: Request):
    """Receive call completion data from Vapi"""
    try:
        payload = await request.json()
        
        # Extract call data
        message = payload.get("message", {})
        call = message.get("call", {})
        call_id = call.get("id")
        
        if not call_id:
            return {"status": "ignored", "reason": "No call ID"}
        
        # Find candidate by call_id
        candidate = await app.mongodb["candidates"].find_one({"vapi_call_id": call_id})
        
        if not candidate:
            print(f"Warning: No candidate found for call_id: {call_id}")
            return {"status": "ignored", "reason": "Candidate not found"}
        
        # Extract call results
        status = message.get("status", "ended")
        transcript = message.get("transcript", "")
        analysis = message.get("analysis", {})
        duration = message.get("duration")  # in seconds
        recording_url = message.get("recordingUrl", "")
        
        # Determine final status based on call outcome
        final_status = "Completed"
        if message.get("endedReason") == "customer-ended-call":
            final_status = "Customer Ended"
        elif message.get("endedReason") == "assistant-error":
            final_status = "Failed"
        
        # Update candidate record
        update_data = {
            "status": final_status,
            "call_ended_at": datetime.utcnow().isoformat(),
            "call_transcript": transcript,
            "call_analysis": analysis,
            "call_recording_url": recording_url
        }
        
        if duration:
            update_data["call_duration"] = duration
        
        await app.mongodb["candidates"].update_one(
            {"_id": candidate["_id"]},
            {"$set": update_data}
        )
        
        print(f"✅ Call completed for {candidate.get('name')} - Duration: {duration}s")
        
        return {"status": "received", "candidate": candidate.get("name")}
        
    except Exception as e:
        print(f"Error processing webhook: {e}")
        return {"status": "error", "message": str(e)}


# 6. WEBHOOK: ASSISTANT REQUEST (for dynamic responses)
@app.post("/webhook/vapi/assistant-request")
async def vapi_assistant_request_webhook(request: Request):
    """Handle dynamic assistant requests during calls"""
    try:
        payload = await request.json()
        
        # You can use this to customize responses mid-call
        # For now, just acknowledge
        return {
            "status": "success",
            "message": "Request received"
        }
        
    except Exception as e:
        print(f"Error in assistant request: {e}")
        return {"status": "error"}


# 7. GET CALL STATUS
@app.get("/api/vapi/call-status/{call_id}")
async def get_call_status(call_id: str, request: Request):
    """Get real-time status of a specific call"""
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    
    candidate = await app.mongodb["candidates"].find_one({"vapi_call_id": call_id})
    
    if not candidate:
        raise HTTPException(status_code=404, detail="Call not found")
    
    return {
        "call_id": call_id,
        "candidate": candidate.get("name"),
        "status": candidate.get("status"),
        "started_at": candidate.get("call_started_at"),
        "ended_at": candidate.get("call_ended_at"),
        "duration": candidate.get("call_duration"),
        "transcript": candidate.get("call_transcript", ""),
        "analysis": candidate.get("call_analysis", {})
    }
