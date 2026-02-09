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
from typing import Optional
import random
import asyncio  # For async delays
import httpx # For Vapi API calls
# 1. Load Config
load_dotenv()

# 2. App Setup
app = FastAPI()

# Add Session Middleware (Required for OAuth)
# WARNING: Ensure SECRET_KEY is set in your .env file
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"), https_only=False, same_site="lax")

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
        return RedirectResponse(url='http://127.0.0.1:5173/dashboard')

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
            "voice": "alloy-openai",
        }
    }

    campaigns_collection = app.mongodb["campaigns"]
    result = await campaigns_collection.insert_one(new_campaign)
    
    new_campaign["_id"] = str(new_campaign["_id"]) 
    new_campaign["id"] = new_campaign["_id"] # Fix: Ensure 'id' matches frontend expectation

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

# --- VAPI CONFIGURATION ---
# Initialize Vapi settings from environment
VAPI_API_KEY = os.getenv("VAPI_API_KEY")
VAPI_PHONE_NUMBER_ID = os.getenv("VAPI_PHONE_NUMBER_ID")

if not VAPI_API_KEY:
    print("⚠️ WARNING: VAPI_API_KEY not found in .env - Calling features will be disabled")
if not VAPI_PHONE_NUMBER_ID:
    print("⚠️ WARNING: VAPI_PHONE_NUMBER_ID not found in .env - Calling features will be disabled")



class BlueprintRequest(BaseModel):
    company_name: str
    job_role: str
    description: str
    candidate_count: int
    agent_persona: str
    strictness: str
    interview_mode: str
    duration: int

@app.post("/api/generate-blueprint")
async def generate_blueprint_api(request: Request):
    try:
        data = await request.json()

        # 1. Extract Data
        company = data.get('company_name', 'TechCorp')
        role = data.get('job_role', 'Candidate')
        mode = data.get('interview_mode', 'technical') # Default to technical
        strictness = data.get('strictness', 'Balanced')
        
        # Deep Context
        professional_domains = data.get('professional_domains', []) # Array of domain IDs
        evaluation_focus = data.get('evaluation_focus', []) # Array of focus IDs
        job_desc = data.get('description', '')
        
        # Agent Persona Settings
        agent_name = data.get('agent_persona', 'Interviewer')
        
        # 2. Build the "Persona Block" (Who is the AI?)
        persona_prompt = f"""
        IDENTITY: You are {agent_name}, a professional AI Recruiter for {company}.
        ROLE: You are interviewing a candidate for the position of {role}.
        """

        # 3. Build the "Behavior Block" based on Strictness
        if "High" in strictness or "hard" in strictness.lower():
            behavior_prompt = "BEHAVIOR: You are skeptical and rigorous. Do not accept vague answers. If the candidate mentions a keyword, ask 'Why?' or 'How?'. Drill down into specific implementation details. If they struggle, move on without helping."
        elif "Low" in strictness or "easy" in strictness.lower():
            behavior_prompt = "BEHAVIOR: You are warm, encouraging, and supportive. If the candidate struggles, offer a small hint. Focus on their potential rather than just right/wrong answers."
        else: # Balanced
            behavior_prompt = "BEHAVIOR: Be professional and neutral. Ask follow-up questions to verify depth, but keep the conversation moving smoothly. Use the STAR method to guide them."

        # 4. Build the "Knowledge Base" (The System Problem/Context)
        knowledge_prompt = ""
        
        # Domain Context
        if professional_domains:
            knowledge_prompt += f"\nPROFESSIONAL DOMAINS: {', '.join(professional_domains)}\n"
            knowledge_prompt += "Focus your technical questions on these domains. Verify expertise in these specific areas.\n"

        # Evaluation Focus
        if evaluation_focus:
            knowledge_prompt += f"\nEVALUATION FOCUS AREAS: {', '.join(evaluation_focus)}\n"
            knowledge_prompt += "Prioritize evaluating the candidate on these specific soft skills and behavioral traits.\n"

        # Mode Specifics
        if mode == 'technical':
            knowledge_prompt += """
            TECHNICAL INTERVIEW STRATEGY:
            - Ask scenario-based questions involving the professional domains.
            - Avoid definition questions (e.g., "What is React?"). Instead ask: "How would you optimize a slow React render cycle?"
            - Evaluate problem-solving skills and depth of knowledge.
            """
            
        if mode == 'hr':
            knowledge_prompt += """
            HR INTERVIEW STRATEGY:
            - Ask behavioral questions using the STAR method (Situation, Task, Action, Result).
            - Focus on cultural fit, communication skills, and the selected evaluation focus areas.
            - Example: "Tell me about a time you handled a conflict with a team member."
            """

        # 5. Final Assembly for Vapi
        system_prompt_for_vapi = f"""
        {persona_prompt}
        
        {behavior_prompt}
        
        CONTEXT FROM JD:
        {job_desc}
        
        {knowledge_prompt}
        
        INTERVIEW GUIDELINES:
        1. Keep responses concise (under 2 sentences).
        2. Wait for the user to finish speaking.
        3. Do not Hallucinate skills the user does not have.
        4. End the interview after gathering sufficient data.
        """

        # 6. Call Groq to structure the output JSON
        # We ask Groq to format this into the JSON structure your UI expects
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an API that generates interview blueprints. Output JSON only."
                },
                {
                    "role": "user",
                    "content": f"""
                    Create a structured interview plan based on this system prompt:
                    {system_prompt_for_vapi}
                    
                    The JSON must contain:
                    1. "system_prompt": The exact text provided above (cleaned up).
                    2. "phases": An array of 3 phases (Intro, Core, Closing).
                    3. "estimated_duration": String.
                    """
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.6,
            response_format={"type": "json_object"},
        )

        response_content = chat_completion.choices[0].message.content
        blueprint = json.loads(response_content)
        
        # Ensure the System Prompt is passed back exactly as we built it (Groq sometimes summarizes it)
        blueprint["system_prompt"] = system_prompt_for_vapi

        return blueprint

    except Exception as e:
        print(f"Error: {e}")
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

@app.put("/api/campaigns/{campaign_id}")
async def update_campaign(campaign_id: str, request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")

    try:
        data = await request.json()
        
        # Prepare update query
        update_fields = {}
        if "config" in data:
            update_fields["config"] = data["config"]
        if "status" in data:
            update_fields["status"] = data["status"]
        if "name" in data:
            update_fields["name"] = data["name"]

        if not update_fields:
            return {"message": "No fields to update"}

        from bson import ObjectId
        result = await app.mongodb["campaigns"].update_one(
            {"_id": ObjectId(campaign_id), "user_id": user["google_id"]},
            {"$set": update_fields}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Campaign not found")

        return {"status": "success", "message": "Campaign updated"}

    except Exception as e:
        print(f"Update Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))





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

# 3. ALL CANDIDATES LIST (For the Candidates Tab - with Campaign Names)
@app.get("/api/candidates/all")
async def get_all_candidates_list(request: Request):
    user = request.session.get('user')
    if not user: return {"candidates": []}

    # Fetch all candidates for this user
    cursor = app.mongodb["candidates"].find({
        "user_id": user["google_id"]
    }).sort("created_at", -1)
    
    candidates = []
    
    # Create a dictionary to cache campaign names (avoid multiple DB calls)
    campaign_cache = {}
    
    async for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        
        # Ensure a default status exists
        if "status" not in doc: 
            doc["status"] = "Pending"
        
        # Fetch campaign name if not in cache
        campaign_id = doc.get("campaign_id")
        if campaign_id and campaign_id not in campaign_cache:
            campaign = await app.mongodb["campaigns"].find_one(
                {"_id": ObjectId(campaign_id)},
                {"name": 1}
            )
            campaign_cache[campaign_id] = campaign.get("name", "Unknown Campaign") if campaign else "Unknown Campaign"
        
        # Add campaign name to candidate
        doc["campaign_name"] = campaign_cache.get(campaign_id, "Unknown Campaign")
        
        candidates.append(doc)
        
    return {"candidates": candidates}


# --- VAPI CALLING LOGIC ---

@app.post("/api/launch-campaign")
async def launch_campaign(request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    
    try:
        data = await request.json()
        campaign_id = data.get('campaign_id')
        
        # 1. Get Campaign & Candidates
        db = app.mongodb
        campaign = await db.campaigns.find_one({"_id": ObjectId(campaign_id)})
        if not campaign: raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Update status to "Running"
        await db.campaigns.update_one(
            {"_id": ObjectId(campaign_id)},
            {"$set": {"status": "Running"}}
        )
        
        # Get pending candidates
        cursor = db.candidates.find({
            "campaign_id": campaign_id,
            "status": "Pending"
        })
        
        candidates = await cursor.to_list(length=100) # Limit batch size
        
        # 2. Launch Calls (Background Task)
        # We start calling immediately but return success to UI
        asyncio.create_task(process_calls(campaign, candidates, data))
        
        return {"status": "success", "message": f"Started calling {len(candidates)} candidates"}
        
    except Exception as e:
        print(f"Launch Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_calls(campaign, candidates, config):
    """Handles the actual Vapi API calls"""
    async with httpx.AsyncClient() as client:
        # Use config passed from frontend or fallback to campaign config
        system_prompt = config.get("system_prompt") or campaign.get("blueprint", {}).get("system_prompt", "You are a helpful interviewer.")
        voice_id = config.get("vapi_voice_id") or campaign.get("config", {}).get("voice", "jennifer-playht")
        
        # Public URL for Webhook (Update this in production)
        SERVER_URL = os.getenv("SERVER_URL") 
        webhook_url = f"{SERVER_URL}/api/vapi-webhook" if SERVER_URL else None

        for candidate in candidates:
            # Check if campaign was stopped
            current_camp = await app.mongodb.campaigns.find_one({"_id": campaign["_id"]})
            if current_camp.get("status") != "Running":
                print("Campaign stopped by user.")
                break
                
            try:
                # Phone formatting
                raw_phone = str(candidate.get("phone", "")).strip()
                if not raw_phone.startswith("+"):
                    raw_phone = "+91" + raw_phone # Default to India for now
                
                # Prepare Vapi Payload
                payload = {
                    "phoneNumberId": VAPI_PHONE_NUMBER_ID,
                    "customer": {
                        "number": raw_phone,
                        "name": candidate.get("name")
                    },
                    "assistant": {
                        "firstMessage": f"Hello {candidate.get('name')}, I am calling from {campaign.get('config', {}).get('company', 'our company')}. Do you have a moment for a quick interview?",
                        "model": {
                            "provider": "openai",
                            "model": "gpt-4",
                            "messages": [
                                {
                                    "role": "system", 
                                    "content": system_prompt
                                }
                            ]
                        },
                        "voice": voice_id,
                        "recordingEnabled": True,
                        "interruptionsEnabled": True,
                        "endCallFunctionEnabled": True,
                        "serverUrl": webhook_url # Receive end-of-call report
                    }
                }
                
                # Make Call
                headers = {
                    "Authorization": f"Bearer {VAPI_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                response = await client.post("https://api.vapi.ai/call", json=payload, headers=headers)
                
                if response.status_code == 201:
                    call_data = response.json()
                    call_id = call_data.get("id")
                    
                    # Update Candidate Status
                    await app.mongodb.candidates.update_one(
                        {"_id": candidate["_id"]},
                        {"$set": {
                            "status": "Dialing", 
                            "call_id": call_id,
                            "last_called": datetime.utcnow().isoformat()
                        }}
                    )
                    print(f"✅ Call queued for {candidate.get('name')} ({call_id})")
                else:
                    print(f"❌ Vapi Error for {candidate.get('name')}: {response.text}")
                    # Mark as Failed
                    await app.mongodb.candidates.update_one(
                        {"_id": candidate["_id"]},
                        {"$set": {"status": "Failed"}}
                    )

            except Exception as e:
                print(f"Call Exception: {e}")
            
            # Respect rate limits / pacing
            await asyncio.sleep(2) 

@app.post("/api/stop-campaign")
async def stop_campaign(request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    
    try:
        data = await request.json()
        campaign_id = data.get('campaign_id')
        
        # 1. Update Status
        await app.mongodb.campaigns.update_one(
            {"_id": ObjectId(campaign_id)},
            {"$set": {"status": "Stopped"}}
        )
        
        return {"status": "success", "message": "Campaign stopped"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- WEBHOOK FOR REPORTING ---
@app.post("/api/vapi-webhook")
async def vapi_webhook(request: Request):
    """Receives End-of-Call Report from Vapi"""
    try:
        data = await request.json()
        message_type = data.get("message", {}).get("type") or data.get("type")
        
        if message_type == "end-of-call-report":
            call_id = data.get("call", {}).get("id")
            analysis = data.get("analysis", {})
            transcript = data.get("transcript", "")
            summary = data.get("summary", "")
            recording_url = data.get("recordingUrl", "")
            
            # Find candidate by call_id
            candidate = await app.mongodb.candidates.find_one({"call_id": call_id})
            
            if candidate:
                # Update Candidate with Report
                await app.mongodb.candidates.update_one(
                    {"_id": candidate["_id"]},
                    {"$set": {
                        "status": "Completed",
                        "report": {
                            "summary": summary,
                            "transcript": transcript,
                            "recording_url": recording_url,
                            "analysis": analysis,
                            "generated_at": datetime.utcnow().isoformat()
                        }
                    }}
                )
                print(f"📄 Report saved for candidate {candidate.get('name')}")
                
        return {"status": "success"}
        
    except Exception as e:
        print(f"Webhook Error: {e}")
        return {"status": "error", "detail": str(e)}









# ==========================================
# 4. CAMPAIGN LAUNCHER (Updates Existing Campaign with Assistant Details)
# ==========================================

# --- NEW PUT ENDPOINT FOR CAMPAIGN UPDATES ---
class UpdateCampaignRequest(BaseModel):
    config: Optional[dict] = None
    status: Optional[str] = None
    name: Optional[str] = None

@app.put("/api/campaigns/{campaign_id}")
async def update_campaign(campaign_id: str, request: UpdateCampaignRequest, req: Request):
    user = req.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        from bson import ObjectId
        query = {"_id": ObjectId(campaign_id), "user_id": user['google_id']}
        
        # Prepare update data
        update_doc = {"$set": {}}
        if request.config:
            update_doc["$set"]["config"] = request.config
        if request.status:
            print(f"Update Status: {request.status} for {campaign_id}")
            update_doc["$set"]["status"] = request.status
        if request.name:
             update_doc["$set"]["name"] = request.name
             
        update_doc["$set"]["updated_at"] = datetime.utcnow().isoformat()

        if not update_doc["$set"]:
             return {"message": "No changes provided"}

        result = await app.mongodb["campaigns"].update_one(query, update_doc)

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        return {"status": "success", "message": "Campaign updated"}
    except Exception as e:
        print(f"Update Campaign Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class CampaignLaunchRequest(BaseModel):
    campaign_id: str  # Changed: Now requires existing campaign ID
    vapi_agent_id: str
    vapi_voice_id: str
    system_prompt: str
    strictness: str
    interview_mode: str

@app.post("/api/launch-campaign")
async def launch_campaign(request: CampaignLaunchRequest, req: Request):
    user = req.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 1. Validate campaign exists and belongs to user
    try:
        campaign = await app.mongodb["campaigns"].find_one({
            "_id": ObjectId(request.campaign_id),
            "user_id": user['google_id']
        })
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found or access denied")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid campaign ID: {str(e)}")

    campaign_name = campaign.get("name", "Unnamed Campaign")
    
    print(f"\n🚀 --- LAUNCHING CAMPAIGN: {campaign_name} ---")
    print(f"📋 Campaign ID: {request.campaign_id}")
    print(f"👤 Agent ID: {request.vapi_agent_id}")
    print(f"🎙️ Voice ID: {request.vapi_voice_id}")
    print(f"🧠 System Prompt Length: {len(request.system_prompt)} chars")
    
    # 2. Update campaign with assistant details and change status to Active
    update_data = {
        "status": "Active",
        "launched_at": datetime.utcnow().isoformat(),
        "config.agent_id": request.vapi_agent_id,
        "config.voice_id": request.vapi_voice_id,
        "config.prompt": request.system_prompt,
        "config.strictness": request.strictness,
        "config.interview_mode": request.interview_mode
    }
    
    await app.mongodb["campaigns"].update_one(
        {"_id": ObjectId(request.campaign_id)},
        {"$set": update_data}
    )
    
    # Verify the update was successful
    updated_campaign = await app.mongodb["campaigns"].find_one({"_id": ObjectId(request.campaign_id)})
    print(f"✅ Campaign {request.campaign_id} updated with assistant details.")
    print(f"🔍 VERIFIED Status in DB: {updated_campaign.get('status', 'NOT SET')}")
    
    # 3. Fetch all candidates for this campaign
    candidates_cursor = app.mongodb["candidates"].find({
        "campaign_id": request.campaign_id
    })
    
    candidates = []
    async for doc in candidates_cursor:
        candidates.append(doc)
    
    candidate_count = len(candidates)
    print(f"👥 Total candidates: {candidate_count}")

    # 4. Check if Vapi is configured
    if not VAPI_API_KEY or not VAPI_PHONE_NUMBER_ID:
        print("⚠️ Vapi not configured - skipping actual calls")
        return {
            "status": "success", 
            "campaign_id": request.campaign_id,
            "campaign_name": campaign_name,
            "campaign_status": updated_campaign.get('status', 'Unknown'),
            "candidate_count": candidate_count,
            "calls_initiated": 0,
            "message": f"Campaign '{campaign_name}' launched (Vapi not configured - no calls made)."
        }
    
    # 5. Make Vapi calls to all candidates
    print(f"\n📞 --- INITIATING CALLS TO {candidate_count} CANDIDATES ---")
    
    calls_initiated = 0
    calls_failed = 0
    
    for idx, candidate in enumerate(candidates, 1):
        candidate_id = str(candidate["_id"])
        candidate_name = candidate.get("name", "Unknown")
        candidate_phone = candidate.get("phone", "")
        
        if not candidate_phone:
            print(f"  ⏭️ Skipping {candidate_name} - No phone number")
            calls_failed += 1
            continue
        
        print(f"\n  📞 [{idx}/{candidate_count}] Calling {candidate_name} at {candidate_phone}...")
        
        # Make the call using Vapi helper
        call_result = vapi_helper.make_outbound_call(
            phone_number=candidate_phone,
            assistant_id=request.vapi_agent_id,
            voice_id=request.vapi_voice_id,
            system_prompt=request.system_prompt,
            phone_number_id=VAPI_PHONE_NUMBER_ID,
            api_key=VAPI_API_KEY,
            use_transient=False  # Use pre-created assistant
        )
        
        if call_result["success"]:
            # Update candidate with call information
            await app.mongodb["candidates"].update_one(
                {"_id": ObjectId(candidate_id)},
                {"$set": {
                    "vapi_call_id": call_result["call_id"],
                    "vapi_assistant_id": request.vapi_agent_id,
                    "vapi_voice_id": request.vapi_voice_id,
                    "call_status": call_result["status"],
                    "call_timestamp": datetime.utcnow().isoformat(),
                    "status": "In Progress"
                }}
            )
            
            print(f"  ✅ Call initiated successfully - Call ID: {call_result['call_id']}")
            calls_initiated += 1
        else:
            # Log the error
            await app.mongodb["candidates"].update_one(
                {"_id": ObjectId(candidate_id)},
                {"$set": {
                    "call_status": "failed",
                    "call_error": call_result.get("error", "Unknown error"),
                    "call_timestamp": datetime.utcnow().isoformat()
                }}
            )
            
            print(f"  ❌ Call failed: {call_result.get('error', 'Unknown error')}")
            calls_failed += 1
        
        # Add small delay to avoid rate limiting (500ms)
        if idx < candidate_count:
            await asyncio.sleep(0.5)
    
    print(f"\n✅ --- CAMPAIGN LAUNCH COMPLETE ---")
    print(f"  📊 Calls Initiated: {calls_initiated}/{candidate_count}")
    print(f"  ❌ Calls Failed: {calls_failed}/{candidate_count}")
    
    return {
        "status": "success", 
        "campaign_id": request.campaign_id,
        "campaign_name": campaign_name,
        "campaign_status": updated_campaign.get('status', 'Unknown'),
        "candidate_count": candidate_count,
        "calls_initiated": calls_initiated,
        "calls_failed": calls_failed,
        "message": f"Campaign '{campaign_name}' launched successfully. {calls_initiated} calls initiated, {calls_failed} failed."
    }


# ==========================================
# 4B. STOP CAMPAIGN (Pause/Stop Active Campaign)
# ==========================================
@app.post("/api/campaigns/{campaign_id}/stop")
async def stop_campaign(campaign_id: str, req: Request):
    user = req.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 1. Validate campaign exists and belongs to user
    try:
        campaign = await app.mongodb["campaigns"].find_one({
            "_id": ObjectId(campaign_id),
            "user_id": user['google_id']
        })
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found or access denied")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid campaign ID: {str(e)}")

    campaign_name = campaign.get("name", "Unnamed Campaign")
    
    print(f"\n⏸️ --- STOPPING CAMPAIGN: {campaign_name} ---")
    print(f"📋 Campaign ID: {campaign_id}")
    
    # 2. Update campaign status to Stopped
    await app.mongodb["campaigns"].update_one(
        {"_id": ObjectId(campaign_id)},
        {"$set": {
            "status": "Stopped",
            "stopped_at": datetime.utcnow().isoformat()
        }}
    )
    
    print(f"✅ Campaign {campaign_id} stopped successfully.")

    return {
        "status": "success", 
        "campaign_id": campaign_id,
        "campaign_name": campaign_name,
        "message": f"Campaign '{campaign_name}' has been stopped."
    }


# ==========================================
# 4C. MANUAL CALL ENDPOINT
# ==========================================
class ManualCallRequest(BaseModel):
    candidate_id: str
    assistant_id: Optional[str] = None
    voice_id: Optional[str] = None
    phone_number: Optional[str] = None  # Override phone number if needed

@app.post("/api/manual-call")
async def make_manual_call(request: ManualCallRequest, req: Request):
    """
    Make a manual call to a specific candidate.
    Can override assistant_id, voice_id, or phone_number.
    """
    user = req.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Check Vapi configuration
    if not VAPI_API_KEY or not VAPI_PHONE_NUMBER_ID:
        raise HTTPException(status_code=503, detail="Vapi is not configured. Please add VAPI_API_KEY and VAPI_PHONE_NUMBER_ID to .env")
    
    # Fetch candidate
    try:
        candidate = await app.mongodb["candidates"].find_one({
            "_id": ObjectId(request.candidate_id),
            "user_id": user['google_id']
        })
        
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found or access denied")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid candidate ID: {str(e)}")
    
    # Get campaign to fetch default assistant_id and voice_id if not provided
    campaign_id = candidate.get("campaign_id")
    campaign = await app.mongodb["campaigns"].find_one({"_id": ObjectId(campaign_id)})
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    # Use provided values or fall back to campaign defaults
    assistant_id = request.assistant_id or campaign.get("config", {}).get("agent_id")
    voice_id = request.voice_id or campaign.get("config", {}).get("voice_id")
    system_prompt = campaign.get("config", {}).get("prompt", "")
    phone_number = request.phone_number or candidate.get("phone", "")
    
    if not phone_number:
        raise HTTPException(status_code=400, detail="No phone number available for this candidate")
    
    if not assistant_id or not voice_id:
        raise HTTPException(status_code=400, detail="Assistant ID or Voice ID not configured")
    
    print(f"\n📞 --- MANUAL CALL ---")
    print(f"👤 Candidate: {candidate.get('name', 'Unknown')}")
    print(f"📱 Phone: {phone_number}")
    print(f"🤖 Assistant ID: {assistant_id}")
    print(f"🎙️ Voice ID: {voice_id}")
    
    # Make the call
    call_result = vapi_helper.make_outbound_call(
        phone_number=phone_number,
        assistant_id=assistant_id,
        voice_id=voice_id,
        system_prompt=system_prompt,
        phone_number_id=VAPI_PHONE_NUMBER_ID,
        api_key=VAPI_API_KEY,
        use_transient=False
    )
    
    if call_result["success"]:
        # Update candidate with call information
        await app.mongodb["candidates"].update_one(
            {"_id": ObjectId(request.candidate_id)},
            {"$set": {
                "vapi_call_id": call_result["call_id"],
                "vapi_assistant_id": assistant_id,
                "vapi_voice_id": voice_id,
                "call_status": call_result["status"],
                "call_timestamp": datetime.utcnow().isoformat(),
                "status": "In Progress"
            }}
        )
        
        print(f"✅ Call initiated successfully - Call ID: {call_result['call_id']}")
        
        return {
            "success": True,
            "message": "Call initiated successfully",
            "call_id": call_result["call_id"],
            "candidate_name": candidate.get("name", "Unknown"),
            "phone_number": phone_number
        }
    else:
        # Log the error
        await app.mongodb["candidates"].update_one(
            {"_id": ObjectId(request.candidate_id)},
            {"$set": {
                "call_status": "failed",
                "call_error": call_result.get("error", "Unknown error"),
                "call_timestamp": datetime.utcnow().isoformat()
            }}
        )
        
        print(f"❌ Call failed: {call_result.get('error', 'Unknown error')}")
        
        raise HTTPException(status_code=500, detail=call_result.get("error", "Failed to initiate call"))


# ==========================================
# 4D. VAPI WEBHOOK ENDPOINT
# ==========================================
@app.post("/api/vapi/webhook")
async def vapi_webhook(request: Request):
    """
    Handle webhook callbacks from Vapi.
    Updates candidate records with call status, transcript, and recording.
    """
    try:
        payload = await request.json()
        
        # Parse webhook data
        webhook_data = vapi_helper.parse_vapi_webhook(payload)
        call_id = webhook_data.get("call_id", "")
        
        if not call_id:
            print("⚠️ Webhook received but no call_id found")
            return {"status": "error", "message": "No call_id in webhook"}
        
        print(f"\n📞 Vapi Webhook Received for Call ID: {call_id}")
        print(f"  Status: {webhook_data.get('status', 'unknown')}")
        print(f"  Duration: {webhook_data.get('duration', 0)}s")
        
        # Find candidate by vapi_call_id
        candidate = await app.mongodb["candidates"].find_one({
            "vapi_call_id": call_id
        })
        
        if not candidate:
            print(f"⚠️ No candidate found for call_id: {call_id}")
            return {"status": "ok", "message": "Candidate not found"}
        
        # Update candidate with webhook data
        update_data = {
            "call_status": webhook_data.get("status", "unknown"),
            "call_duration": webhook_data.get("duration", 0),
            "call_recording_url": webhook_data.get("recording_url", ""),
            "call_end_reason": webhook_data.get("end_reason", ""),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # If call is completed, update transcript
        if webhook_data.get("transcript"):
            update_data["call_transcript"] = webhook_data["transcript"]
        
        # Update candidate status based on call status
        call_status = webhook_data.get("status", "")
        if call_status == "ended":
            # Call completed - keep current status or mark as completed
            update_data["status"] = candidate.get("status", "Completed")
        elif call_status in ["failed", "busy", "no-answer"]:
            update_data["status"] = "Failed"
            update_data["call_error"] = webhook_data.get("end_reason", "Call failed")
        
        await app.mongodb["candidates"].update_one(
            {"_id": candidate["_id"]},
            {"$set": update_data}
        )
        
        print(f"✅ Updated candidate: {candidate.get('name', 'Unknown')}")
        
        return {"status": "ok", "message": "Webhook processed successfully"}
        
    except Exception as e:
        print(f"❌ Error processing webhook: {str(e)}")
        return {"status": "error", "message": str(e)}


# ==========================================
# 5. DEBUG ENDPOINT (Check Campaign Configuration)
# ==========================================
@app.get("/api/debug/campaign")
async def get_campaign_debug(campaign_id: str = None, request: Request = None):
    """
    Debug endpoint to inspect campaign configuration.
    Usage: 
    - /api/debug/campaign?campaign_id=<id>  -> Get specific campaign
    - /api/debug/campaign                   -> Get latest campaign
    """
    
    if campaign_id:
        # Fetch specific campaign by ID
        try:
            campaign = await app.mongodb["campaigns"].find_one(
                {"_id": ObjectId(campaign_id)}
            )
            if not campaign:
                return {"status": "error", "message": f"Campaign with ID '{campaign_id}' not found."}
        except Exception as e:
            return {"status": "error", "message": f"Invalid campaign ID: {str(e)}"}
    else:
        # Fetch the latest campaign
        campaign = await app.mongodb["campaigns"].find_one(
            sort=[("created_at", -1)]
        )
        
        if not campaign:
            return {"status": "No campaigns found in database."}

    # Count candidates for this campaign
    campaign_id_str = str(campaign["_id"])
    candidate_count = await app.mongodb["candidates"].count_documents({
        "campaign_id": campaign_id_str
    })

    # Build detailed response
    return {
        "status": "success",
        "campaign_id": campaign_id_str,
        "campaign_name": campaign.get("name", "Unnamed"),
        "campaign_status": campaign.get("status", "Unknown"),
        "created_at": campaign.get("created_at"),
        "launched_at": campaign.get("launched_at", "Not launched yet"),
        "candidate_count": candidate_count,
        "configuration": {
            "agent_id": campaign.get("config", {}).get("agent_id", "NOT SET"),
            "voice_id": campaign.get("config", {}).get("voice_id", "NOT SET"),
            "strictness": campaign.get("config", {}).get("strictness", "NOT SET"),
            "interview_mode": campaign.get("config", {}).get("interview_mode", "NOT SET"),
            "system_prompt_length": len(campaign.get("config", {}).get("prompt", "")) if campaign.get("config", {}).get("prompt") else 0,
            "system_prompt_preview": (campaign.get("config", {}).get("prompt", "NOT SET")[:200] + "...") if campaign.get("config", {}).get("prompt") else "NOT SET"
        }
    }

# ==========================================
# 6. CANDIDATE INTERVIEW REPORT ENDPOINT
# ==========================================

def generate_mock_interview_data(candidate_name: str, candidate_email: str, candidate_phone: str):
    """Generate realistic mock interview data for testing"""
    
    # Generate random scores
    overall_score = round(random.uniform(4.0, 9.5), 1)
    confidence = round(random.uniform(5.0, 9.5), 1)
    communication = round(random.uniform(5.0, 9.0), 1)
    technical = round(random.uniform(4.0, 9.5), 1)
    problem_solving = round(random.uniform(5.0, 9.0), 1)
    cultural_fit = round(random.uniform(5.5, 9.0), 1)
    
    # Sample transcript based on score level
    if overall_score >= 7.5:
        transcript = [
            {"timestamp": "00:00:05", "speaker": "AI", "text": f"Hello {candidate_name.split()[0]}, thank you for joining us today. Can you tell me about your experience with React?"},
            {"timestamp": "00:00:12", "speaker": "Candidate", "text": "Yes, I have been working with React for about 3 years now. I've built several production applications using React with Redux for state management."},
            {"timestamp": "00:00:28", "speaker": "AI", "text": "That's great. Can you walk me through a challenging problem you faced and how you solved it?"},
            {"timestamp": "00:00:35", "speaker": "Candidate", "text": "Sure, we had a performance issue with our dashboard that was rendering thousands of rows. I implemented virtualization using react-window which reduced the render time from 8 seconds to under 1 second."},
            {"timestamp": "00:01:05", "speaker": "AI", "text": "Excellent solution. How do you handle state management in large applications?"},
            {"timestamp": "00:01:12", "speaker": "Candidate", "text": "I prefer using Redux Toolkit for complex state, but for simpler cases, I use Context API with useReducer. It really depends on the application's complexity."},
            {"timestamp": "00:01:35", "speaker": "AI", "text": "How do you approach testing your React components?"},
            {"timestamp": "00:01:42", "speaker": "Candidate", "text": "I use Jest and React Testing Library. I focus on testing user interactions rather than implementation details. We maintain about 80% test coverage."},
        ]
        strengths = [
            "Strong technical foundation with 3+ years of React experience",
            "Demonstrated problem-solving skills with concrete examples",
            "Good understanding of performance optimization",
            "Clear and articulate communication",
            "Follows best practices in testing and state management"
        ]
        weaknesses = [
            "Could improve knowledge of newer React features like Server Components",
            "Limited experience with TypeScript mentioned"
        ]
        insights = "Excellent candidate with strong React expertise. Shows practical problem-solving ability and clear communication. Recommended for next round."
    else:
        transcript = [
            {"timestamp": "00:00:05", "speaker": "AI", "text": f"Hello {candidate_name.split()[0]}, thank you for joining us. Can you tell me about your experience with React?"},
            {"timestamp": "00:00:12", "speaker": "Candidate", "text": "Um, yes, I've used React. I learned it in a bootcamp last year."},
            {"timestamp": "00:00:22", "speaker": "AI", "text": "Can you explain how React hooks work?"},
            {"timestamp": "00:00:28", "speaker": "Candidate", "text": "Hooks are... um... functions that let you use state? Like useState and useEffect."},
            {"timestamp": "00:00:40", "speaker": "AI", "text": "Can you give me an example of when you'd use useEffect?"},
            {"timestamp": "00:00:48", "speaker": "Candidate", "text": "When you want to... fetch data? Or do something when the component loads."},
            {"timestamp": "00:01:02", "speaker": "AI", "text": "How do you handle errors in your React applications?"},
            {"timestamp": "00:01:10", "speaker": "Candidate", "text": "I use try-catch blocks. Sometimes console.log to check for errors."},
        ]
        strengths = [
            "Shows basic understanding of React fundamentals",
            "Willing to learn and improve",
            "Polite and professional demeanor"
        ]
        weaknesses = [
            "Limited practical experience with React",
            "Struggles to articulate technical concepts clearly",
            "Needs more hands-on project experience",
            "Limited knowledge of advanced React patterns"
        ]
        insights = "Candidate has basic React knowledge but lacks depth. Would benefit from more practical experience before taking on senior roles."
    
    return {
        "candidate": {
            "name": candidate_name,
            "email": candidate_email,
            "phone": candidate_phone
        },
        "interview": {
            "date": datetime.utcnow().isoformat(),
            "duration": random.randint(480, 1200),  # 8-20 minutes in seconds
            "status": "completed",
            "recording_url": f"/recordings/interview_{random.randint(1000, 9999)}.mp3"
        },
        "transcript": transcript,
        "scores": {
            "overall": overall_score,
            "confidence": confidence,
            "communication": communication,
            "technical_knowledge": technical,
            "problem_solving": problem_solving,
            "cultural_fit": cultural_fit
        },
        "analysis": {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "key_insights": insights,
            "hesitation_count": random.randint(3, 15),
            "filler_words_count": random.randint(5, 25),
            "average_response_time": round(random.uniform(2.0, 5.5), 1)
        }
    }


@app.get("/api/campaigns/{campaign_id}/candidate/{candidate_id}/report")
async def get_candidate_interview_report(campaign_id: str, candidate_id: str, request: Request):
    """
    Get detailed interview report for a specific candidate.
    Returns mock data for now - will be replaced with real interview data later.
    """
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    
    try:
        # 1. Fetch candidate from database
        candidate = await app.mongodb["candidates"].find_one({
            "_id": ObjectId(candidate_id),
            "campaign_id": campaign_id
        })
        
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        # 2. Check if real interview data exists (for future integration)
        # For now, we'll always generate mock data
        interview_results_coll = app.mongodb.get("interview_results")
        if interview_results_coll:
            interview_results = await interview_results_coll.find_one({
                "candidate_id": candidate_id,
                "campaign_id": campaign_id
            })
            
            if interview_results:
                # Return real interview data if it exists
                interview_results["id"] = str(interview_results.pop("_id"))
                return {"success": True, "data": interview_results}
        
        # Generate mock data for testing
        mock_data = generate_mock_interview_data(
            candidate.get("name", "Unknown Candidate"),
            candidate.get("email", ""),
            candidate.get("phone", "")
        )
        
        return {"success": True, "data": mock_data}
    
    except Exception as e:
        print(f"Error fetching interview report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch interview report: {str(e)}")



@app.get("/api/candidates/{candidate_id}/report")
async def get_candidate_report(candidate_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)

    try:
        # 1. Fetch Candidate Basic Info
        db = app.mongodb["candidates"]
        candidate = await db.find_one({"_id": ObjectId(candidate_id)})
        
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
            
        candidate["id"] = str(candidate.pop("_id"))

        # 2. Generate MOCK Interview Data (Simulating the AI Engine result)
        # In a real scenario, you would fetch this from an 'interviews' collection
        
        mock_scores = {
            "confidence": random.randint(70, 98),
            "communication": random.randint(65, 95),
            "technical": random.randint(60, 92),
            "cultural": random.randint(75, 99)
        }
        
        # Calculate weighted average
        total_score = int((mock_scores["confidence"] + mock_scores["communication"]*1.2 + mock_scores["technical"]*1.5 + mock_scores["cultural"]) / 4.7)

        # 🎯 AUTO-SELECTION LOGIC: Based on AI Score
        # Business Rule: Score >= 70 = Selected, < 70 = Rejected
        auto_status = "Selected" if total_score >= 70 else "Rejected"
        
        # Update candidate status in database
        await db.update_one(
            {"_id": ObjectId(candidate_id)},
            {"$set": {
                "status": auto_status,
                "ai_score": total_score,
                "updated_at": datetime.utcnow().isoformat()
            }}
        )

        mock_transcript = [
            {"role": "ai", "text": "Hello, thank you for joining. Let's start with your experience in Python.", "time": "00:05"},
            {"role": "user", "text": "Hi! Yes, I've been using Python for about 4 years now, mostly for backend development using FastAPI and Django.", "time": "00:12"},
            {"role": "ai", "text": "That's great. Can you explain how you handle database migrations in a production environment?", "time": "00:25"},
            {"role": "user", "text": "I usually stick to the standard ORM tools. For Django, I use manage.py migrate. However, for large datasets, I ensure to lock tables minimally or use tools like gh-ost if it's MySQL.", "time": "00:38"},
            {"role": "ai", "text": "Excellent detailed answer. Now, tell me about a time you faced a difficult bug.", "time": "00:50"},
            {"role": "user", "text": "We had a memory leak in one of our microservices. I used a profiler to trace it back to an unclosed file handler in a utility function.", "time": "01:15"},
            {"role": "ai", "text": "Very impressive problem solving. One last question about cultural fit...", "time": "01:30"}
        ]
        
        return {
            "candidate": candidate,
            "interview_data": {
                "status": auto_status,
                "duration": "14m 30s",
                "date": datetime.utcnow().isoformat(),
                "overall_score": total_score,
                "scores": mock_scores,
                "transcript": mock_transcript,
                "summary": f"Candidate automatically {auto_status.lower()} based on AI score of {total_score}/100. " + 
                          ("Strong technical knowledge and communication skills demonstrated." if auto_status == "Selected" else "Needs improvement in technical areas.")
            }
        }
    except Exception as e:
        print(f"Report Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# CSV EXPORT ENDPOINTS
# ==========================================
from fastapi.responses import StreamingResponse
import csv
from io import StringIO

@app.get("/api/campaigns/{campaign_id}/export/all")
async def export_all_candidates(campaign_id: str, request: Request):
    """Export all candidates for a campaign as CSV"""
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    
    try:
        # Fetch all candidates for this campaign
        db = app.mongodb["candidates"]
        cursor = db.find({"campaign_id": campaign_id})
        
        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(["Name", "Email", "Phone", "Status", "AI Score", "Date Added"])
        
        # Write data rows
        async for candidate in cursor:
            writer.writerow([
                candidate.get("name", ""),
                candidate.get("email", ""),
                candidate.get("phone", ""),
                candidate.get("status", "Pending"),
                candidate.get("ai_score", "N/A"),
                candidate.get("created_at", "")[:10] if candidate.get("created_at") else ""
            ])
        
        # Prepare response
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=all_candidates_{campaign_id}.csv"}
        )
        
    except Exception as e:
        print(f"Export Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/campaigns/{campaign_id}/export/selected")
async def export_selected_candidates(campaign_id: str, request: Request):
    """Export only selected candidates for a campaign as CSV"""
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    
    try:
        # Fetch only Selected candidates
        db = app.mongodb["candidates"]
        cursor = db.find({
            "campaign_id": campaign_id,
            "status": "Selected"
        })
        
        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(["Name", "Email", "Phone", "AI Score", "Date Selected"])
        
        # Write data rows
        async for candidate in cursor:
            writer.writerow([
                candidate.get("name", ""),
                candidate.get("email", ""),
                candidate.get("phone", ""),
                candidate.get("ai_score", "N/A"),
                candidate.get("updated_at", "")[:10] if candidate.get("updated_at") else ""
            ])
        
        # Prepare response
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=selected_candidates_{campaign_id}.csv"}
        )
        
    except Exception as e:
        print(f"Export Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# CAMPAIGN STATISTICS ENDPOINT
# ==========================================

@app.get("/api/campaigns/{campaign_id}/statistics")
async def get_campaign_statistics(campaign_id: str, request: Request):
    """Get real-time campaign statistics for performance dashboard"""
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    
    try:
        db = app.mongodb["candidates"]
        
        # Count candidates by status
        total = await db.count_documents({"campaign_id": campaign_id})
        selected = await db.count_documents({"campaign_id": campaign_id, "status": "Selected"})
        rejected = await db.count_documents({"campaign_id": campaign_id, "status": "Rejected"})
        pending = await db.count_documents({"campaign_id": campaign_id, "status": {"$in": ["Pending", "In Progress", "Scheduled"]}})
        
        # Calculate percentages for success rate analysis
        selected_pct = round((selected / total * 100), 1) if total > 0 else 0
        rejected_pct = round((rejected / total * 100), 1) if total > 0 else 0
        pending_pct = round((pending / total * 100), 1) if total > 0 else 0
        
        # Calculate trend (comparing to previous period - mock for now)
        # In production, you'd compare to last week's data
        selected_trend = "+12%" if selected > 0 else "0%"
        rejected_trend = "-5%" if rejected < pending else "+8%"
        
        return {
            "total": total,
            "selected": {
                "count": selected,
                "trend": selected_trend,
                "percentage": selected_pct
            },
            "rejected": {
                "count": rejected,
                "trend": rejected_trend,
                "percentage": rejected_pct
            },
            "pending": {
                "count": pending,
                "status": "Processing",
                "percentage": pending_pct
            },
            "success_rate": {
                "selected_percent": selected_pct,
                "pending_percent": pending_pct,
                "rejected_percent": rejected_pct
            }
        }
        
    except Exception as e:
        print(f"Statistics Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

