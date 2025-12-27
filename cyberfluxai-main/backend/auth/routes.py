from fastapi import APIRouter, HTTPException, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime, timezone, timedelta
import logging

from auth.models import UserRegister, UserLogin, Token, UserResponse, User
from auth.utils import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_active_user
)

logger = logging.getLogger(__name__)

auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# Database will be injected
db = None


def set_database(database: AsyncIOMotorDatabase):
    """Set database instance"""
    global db
    db = database


@auth_router.post("/register", response_model=UserResponse)
async def register(user_data: UserRegister):
    """Register new user"""
    try:
        # Check if user already exists
        existing_user = await db.users.find_one({
            "$or": [
                {"email": user_data.email},
                {"username": user_data.username}
            ]
        })
        
        if existing_user:
            if existing_user.get("email") == user_data.email:
                raise HTTPException(status_code=400, detail="Email already registered")
            else:
                raise HTTPException(status_code=400, detail="Username already taken")
        
        # Hash password
        hashed_password = get_password_hash(user_data.password)
        
        # Create user
        user = User(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            is_active=True,
            is_admin=False,
            created_at=datetime.now(timezone.utc)
        )
        
        # Save to database
        user_dict = user.model_dump()
        await db.users.insert_one(user_dict)
        
        logger.info(f"New user registered: {user_data.username}")
        
        return UserResponse(
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            is_admin=user.is_admin,
            created_at=user.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@auth_router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    """Login user and return JWT token"""
    try:
        # Find user
        user = await db.users.find_one({"username": credentials.username})
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        # Verify password
        if not verify_password(credentials.password, user["hashed_password"]):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        # Check if user is active
        if not user.get("is_active", True):
            raise HTTPException(status_code=401, detail="User account is disabled")
        
        # Update last login
        await db.users.update_one(
            {"username": credentials.username},
            {"$set": {"last_login": datetime.now(timezone.utc)}}
        )
        
        # Create access token
        access_token = create_access_token(
            data={
                "sub": user["username"],
                "email": user["email"],
                "is_admin": user.get("is_admin", False)
            },
            expires_delta=timedelta(hours=24)
        )
        
        logger.info(f"User logged in: {credentials.username}")
        
        return Token(access_token=access_token, token_type="bearer")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@auth_router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_active_user)):
    """Get current user information"""
    try:
        user = await db.users.find_one({"username": current_user["username"]}, {"_id": 0, "hashed_password": 0})
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(
            email=user["email"],
            username=user["username"],
            full_name=user["full_name"],
            is_admin=user.get("is_admin", False),
            created_at=user["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user info error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user information")


@auth_router.post("/logout")
async def logout(current_user: dict = Depends(get_current_active_user)):
    """Logout user (client should delete token)"""
    logger.info(f"User logged out: {current_user['username']}")
    return {"message": "Successfully logged out"}