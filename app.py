from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import logging
from typing import List, Dict, Optional
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Car Recommendation System", version="2.0")

# Get CORS origins from environment variable
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Configuration
class Config:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    TOP_K_RECOMMENDATIONS = int(os.getenv("TOP_K_RECOMMENDATIONS", "10"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    CACHE_EMBEDDINGS = os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true"
    CACHE_FILE = os.getenv("CACHE_FILE", "car_embeddings_cache.pkl")
    LOG_FILE = os.getenv("LOG_FILE", "recommendation_advanced.log")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

config = Config()

# Initialize model
print(f"Loading model: {config.EMBEDDING_MODEL}")
model = SentenceTransformer(config.EMBEDDING_MODEL)

# Logging setup
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class CarRecommendationSystem:
    def __init__(self):
        self.df = None
        self.car_embeddings = None
        self.scaler = StandardScaler()
        self.numerical_features = None
        self.model = model  # Attach the global model to the instance
        self.load_data()
        self.preprocess_data()
        self.create_embeddings()
    
    def load_data(self):
        """Load and validate car dataset"""
        try:
            self.df = pd.read_csv("cars.csv")
            logger.info(f"Loaded {len(self.df)} cars from dataset")
        except FileNotFoundError:
            logger.error("cars.csv not found. Please ensure the file exists.")
            raise HTTPException(status_code=500, detail="Car dataset not found")
    
    def preprocess_data(self):
        """Enhanced data preprocessing with better handling"""
        # Fill missing values with appropriate defaults
        self.df = self.df.fillna({
            'Make': 'Unknown',
            'Model': 'Unknown',
            'Variant': 'Standard',
            'Body_Type': 'Unknown',
            'Fuel_Type': 'Unknown',
            'Power': '0',
            'Torque': '0',
            'Price': '0',
            'ARAI_Certified_Mileage': '0',
            'Seating_Capacity': '5',
            'Transmission': 'Manual'
        })
        
        # Extract numerical values for hybrid scoring
        self.extract_numerical_features()
        
        # Create rich descriptions
        self.df["description"] = self.df.apply(self.create_enhanced_description, axis=1)
        self.df["search_text"] = self.df.apply(self.create_search_text, axis=1)
        
        logger.info("Data preprocessing completed")
    
    def extract_numerical_features(self):
        """Extract numerical features for hybrid recommendation"""
        def extract_price(text):
            if pd.isna(text):
                return 0
            # Extract price in rupees and convert to lakhs
            text = str(text).lower()
            # Remove commas and spaces
            text = re.sub(r'[,\s]', '', text)
            # Extract numbers
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if not numbers:
                return 0
            price = float(numbers[0])
            
            # Convert to lakhs based on format
            if 'crore' in text:
                return price * 100  # Convert crores to lakhs
            elif 'lakh' in text:
                return price
            elif len(numbers[0]) >= 6:  # If number is >= 6 digits, assume it's in rupees
                return price / 100000  # Convert rupees to lakhs
            else:
                return price  # Assume already in lakhs
        
        def extract_number(text):
            if pd.isna(text):
                return 0
            # Extract first number from string
            numbers = re.findall(r'\d+(?:\.\d+)?', str(text))
            return float(numbers[0]) if numbers else 0
        
        # Extract numerical values
        self.df['power_num'] = self.df['Power'].apply(extract_number)
        self.df['price_num'] = self.df['Price'].apply(extract_price)
        self.df['mileage_num'] = self.df['ARAI_Certified_Mileage'].apply(extract_number)
        self.df['seating_num'] = self.df['Seating_Capacity'].apply(extract_number)
        
        # Prepare numerical features for scaling
        self.numerical_features = ['power_num', 'price_num', 'mileage_num', 'seating_num']
        feature_matrix = self.df[self.numerical_features].values
        self.scaled_features = self.scaler.fit_transform(feature_matrix)
    
    def create_enhanced_description(self, row):
        """Create a comprehensive, natural language description"""
        # Base description
        desc_parts = []
        
        # Car identity
        identity = f"{row.get('Make', '')} {row.get('Model', '')} {row.get('Variant', '')}"
        desc_parts.append(f"The {identity.strip()}")
        
        # Specifications
        if row.get('Body_Type'):
            desc_parts.append(f"is a {row.get('Body_Type')} vehicle")
        
        if row.get('Fuel_Type'):
            desc_parts.append(f"powered by {row.get('Fuel_Type')} engine")
        
        if row.get('Power'):
            desc_parts.append(f"delivering {row.get('Power')} power")
        
        if row.get('Torque'):
            desc_parts.append(f"with {row.get('Torque')} torque")
        
        # Performance and comfort
        if row.get('ARAI_Certified_Mileage'):
            desc_parts.append(f"offering {row.get('ARAI_Certified_Mileage')} fuel efficiency")
        
        if row.get('Seating_Capacity'):
            desc_parts.append(f"seating {row.get('Seating_Capacity')} people")
        
        if row.get('Transmission'):
            desc_parts.append(f"with {row.get('Transmission')} transmission")
        
        # Price
        if row.get('Price'):
            desc_parts.append(f"priced at {row.get('Price')}")
        
        # Join all parts
        description = " ".join(desc_parts).strip()
        
        # Add category keywords for better matching
        category_keywords = self.get_category_keywords(row)
        if category_keywords:
            description += f" Keywords: {category_keywords}"
        
        return self.clean_text(description)
    
    def create_search_text(self, row):
        """Create optimized search text for better matching"""
        search_parts = []
        
        # Basic info
        search_parts.extend([
            str(row.get('Make', '')),
            str(row.get('Model', '')),
            str(row.get('Variant', ''))
        ])
        
        # Categories and types
        search_parts.extend([
            str(row.get('Body_Type', '')),
            str(row.get('Fuel_Type', '')),
            str(row.get('Transmission', ''))
        ])
        
        # Add synonyms and alternative terms
        search_parts.extend(self.get_synonyms(row))
        
        return self.clean_text(" ".join(search_parts))
    
    def get_category_keywords(self, row):
        """Generate category-specific keywords"""
        keywords = []
        
        # Body type categories
        body_type = str(row.get('Body_Type', '')).lower()
        if 'suv' in body_type:
            keywords.extend(['suv', 'sport utility', 'off-road', 'spacious'])
        elif 'sedan' in body_type:
            keywords.extend(['sedan', 'comfort', 'luxury', 'family'])
        elif 'hatchback' in body_type:
            keywords.extend(['hatchback', 'compact', 'city', 'economical'])
        elif 'coupe' in body_type:
            keywords.extend(['coupe', 'sporty', 'stylish', 'performance'])
        
        # Fuel type categories
        fuel_type = str(row.get('Fuel_Type', '')).lower()
        if 'petrol' in fuel_type:
            keywords.extend(['petrol', 'gasoline', 'performance'])
        elif 'diesel' in fuel_type:
            keywords.extend(['diesel', 'fuel efficient', 'torque'])
        elif 'electric' in fuel_type:
            keywords.extend(['electric', 'ev', 'eco-friendly', 'zero emission'])
        elif 'hybrid' in fuel_type:
            keywords.extend(['hybrid', 'fuel efficient', 'eco-friendly'])
        
        # Price categories
        price = row.get('price_num', 0)
        if price > 0:
            if price < 500000:
                keywords.append('budget')
            elif price < 1000000:
                keywords.append('mid-range')
            elif price < 2000000:
                keywords.append('premium')
            else:
                keywords.append('luxury')
        
        return " ".join(keywords)
    
    def get_synonyms(self, row):
        """Add synonyms for better matching"""
        synonyms = []
        
        # Common synonyms
        synonyms_dict = {
            'suv': ['sport utility vehicle', 'crossover'],
            'sedan': ['saloon', 'four-door'],
            'hatchback': ['five-door', 'compact'],
            'petrol': ['gasoline', 'gas'],
            'diesel': ['oil', 'compression ignition'],
            'automatic': ['auto', 'cvt', 'amt'],
            'manual': ['stick shift', 'gear']
        }
        
        for key, value_list in synonyms_dict.items():
            for field in ['Body_Type', 'Fuel_Type', 'Transmission']:
                if key in str(row.get(field, '')).lower():
                    synonyms.extend(value_list)
        
        return synonyms
    
    def clean_text(self, text):
        """Enhanced text cleaning"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s.,\-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_embeddings(self):
        """Create or load embeddings for all cars"""
        if config.CACHE_EMBEDDINGS and os.path.exists(config.CACHE_FILE):
            with open(config.CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
                self.car_embeddings = cache["embeddings"]
                logger.info("Loaded car embeddings from cache.")
        else:
            descriptions = self.df["description"].tolist()
            self.car_embeddings = self.model.encode(descriptions, convert_to_tensor=True)
            if config.CACHE_EMBEDDINGS:
                with open(config.CACHE_FILE, "wb") as f:
                    pickle.dump({"embeddings": self.car_embeddings}, f)
                logger.info("Saved car embeddings to cache.")
    
    def hybrid_similarity(self, query_embedding, numerical_weights=None):
        """Compute hybrid similarity using self.model"""
        if numerical_weights is None:
            numerical_weights = {'power': 0.2, 'price': 0.3, 'mileage': 0.3, 'seating': 0.2}
        
        # Semantic similarity
        semantic_scores = util.cos_sim(query_embedding, self.car_embeddings)[0]
        
        # For now, return semantic scores (can be enhanced with user preferences)
        return semantic_scores
    
    def get_match_confidence(self, score):
        """Convert similarity score to confidence level"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def get_recommendations(self, query: str, user_preferences: Dict = None):
        """Get enhanced recommendations with constraint-based filtering"""
        # Extract constraints from query
        constraints = self.extract_query_constraints(query)
        
        # Apply hard filters first
        valid_indices = self.apply_hard_filters(constraints)
        
        if not valid_indices:
            # If no cars match constraints, relax some filters
            logger.warning(f"No cars match strict constraints for query: {query}")
            # Try relaxing price constraint by 20%
            if constraints['max_price'] is not None:
                constraints['max_price'] *= 1.2
                valid_indices = self.apply_hard_filters(constraints)
            
            if not valid_indices:
                # If still no matches, use all cars but penalize mismatches
                valid_indices = list(range(len(self.df)))
        
        # Clean and encode query
        clean_query = self.clean_text(query)
        query_embedding = self.model.encode(clean_query, convert_to_tensor=True)
        
        # Get similarity scores only for valid indices
        if len(valid_indices) < len(self.car_embeddings):
            valid_embeddings = self.car_embeddings[valid_indices]
            similarity_scores = util.cos_sim(query_embedding, valid_embeddings)[0]
        else:
            similarity_scores = util.cos_sim(query_embedding, self.car_embeddings)[0]
            valid_indices = list(range(len(self.df)))
        
        # Apply constraint-based boosting
        boosted_scores = self.apply_constraint_boosting(
            similarity_scores, valid_indices, constraints
        )
        
        # Get top recommendations
        top_k = min(config.TOP_K_RECOMMENDATIONS, len(boosted_scores))
        if top_k == 0:
            return []
        
        top_indices = torch.topk(boosted_scores, k=top_k)
        
        recommendations = []
        for i, score_idx in enumerate(top_indices.indices.tolist()):
            # Map back to original dataframe index
            original_idx = valid_indices[score_idx]
            score = top_indices.values[i].item()
            car = self.df.iloc[original_idx]
            
            # Calculate constraint satisfaction
            constraint_satisfaction = self.calculate_constraint_satisfaction(car, constraints)
            
            recommendation = {
                "rank": i + 1,
                "Make": str(car.get("Make", "")),
                "Model": str(car.get("Model", "")),
                "Variant": str(car.get("Variant", "")),
                "Body_Type": str(car.get("Body_Type", "")),
                "Fuel_Type": str(car.get("Fuel_Type", "")),
                "Power": str(car.get("Power", "")),
                "Torque": str(car.get("Torque", "")),
                "Price": str(car.get("Price", "")),
                "ARAI_Certified_Mileage": str(car.get("ARAI_Certified_Mileage", "")),
                "Seating_Capacity": str(car.get("Seating_Capacity", "")),
                "Transmission": str(car.get("Transmission", "")),
                "similarity_score": float(score),
                "match_confidence": self.get_match_confidence(score),
                "constraint_satisfaction": constraint_satisfaction,
                "price_lakhs": float(car.get("price_num", 0)),
                "description": str(car.get("description", "")),
                "why_recommended": self.generate_recommendation_reason(car, constraints)
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def apply_constraint_boosting(self, similarity_scores, valid_indices, constraints):
        """Apply boosting based on how well cars match constraints"""
        boosted_scores = similarity_scores.clone()
        
        for i, idx in enumerate(valid_indices):
            car = self.df.iloc[idx]
            boost_factor = 1.0
            
            # Price constraint boosting
            if constraints['max_price'] is not None:
                car_price = car.get('price_num', 0)
                if car_price <= constraints['max_price']:
                    # Boost cars well within budget
                    price_ratio = car_price / constraints['max_price']
                    if price_ratio <= 0.8:  # Well within budget
                        boost_factor += 0.15
                    elif price_ratio <= 0.9:  # Close to budget
                        boost_factor += 0.1
                else:
                    # Penalize cars over budget
                    boost_factor -= 0.3
            
            # Fuel efficiency boosting for budget queries
            if constraints['budget_category'] == 'budget':
                mileage = car.get('mileage_num', 0)
                if mileage >= 15:  # Good fuel efficiency
                    boost_factor += 0.1
            
            # Seating capacity exact match boosting
            if constraints['min_seating'] is not None:
                car_seating = car.get('seating_num', 0)
                if car_seating == constraints['min_seating']:
                    boost_factor += 0.1
            
            boosted_scores[i] *= boost_factor
        
        return boosted_scores
    
    def calculate_constraint_satisfaction(self, car, constraints):
        """Calculate how well a car satisfies the constraints"""
        satisfaction = {}
        
        if constraints['max_price'] is not None:
            car_price = car.get('price_num', 0)
            satisfaction['price'] = car_price <= constraints['max_price']
            satisfaction['price_value'] = f"₹{car_price:.1f}L (Budget: ₹{constraints['max_price']:.1f}L)"
        
        if constraints['min_seating'] is not None:
            car_seating = car.get('seating_num', 0)
            satisfaction['seating'] = car_seating >= constraints['min_seating']
            satisfaction['seating_value'] = f"{car_seating} seats (Required: {constraints['min_seating']}+)"
        
        if constraints['fuel_type'] is not None:
            car_fuel = str(car.get('Fuel_Type', '')).lower()
            satisfaction['fuel_type'] = constraints['fuel_type'] in car_fuel
            satisfaction['fuel_type_value'] = car_fuel
        
        return satisfaction
    
    def generate_recommendation_reason(self, car, constraints):
        """Generate explanation for why this car is recommended"""
        reasons = []
        
        if constraints['max_price'] is not None:
            car_price = car.get('price_num', 0)
            if car_price <= constraints['max_price']:
                reasons.append(f"Within budget at ₹{car_price:.1f}L")
            else:
                reasons.append(f"Slightly over budget at ₹{car_price:.1f}L")
        
        # Fuel efficiency
        mileage = car.get('mileage_num', 0)
        if mileage >= 15:
            reasons.append(f"Good fuel efficiency ({mileage} km/l)")
        
        # Seating
        if constraints['min_seating'] is not None:
            car_seating = car.get('seating_num', 0)
            if car_seating >= constraints['min_seating']:
                reasons.append(f"Adequate seating for {car_seating} people")
        
        # Body type match
        if constraints['body_type'] is not None:
            body_type = str(car.get('Body_Type', '')).lower()
            if constraints['body_type'] in body_type:
                reasons.append(f"Matches {constraints['body_type']} requirement")
        
        return "; ".join(reasons) if reasons else "Good overall match"
    
    def extract_query_constraints(self, query: str):
        """Extract price, seating, and other constraints from query"""
        constraints = {
            'max_price': None,
            'min_price': None,
            'min_seating': None,
            'max_seating': None,
            'fuel_type': None,
            'body_type': None,
            'transmission': None,
            'budget_category': None
        }
        
        query_lower = query.lower()
        
        # Price extraction patterns
        price_patterns = [
            # "under X lakhs/crores"
            (r'under\s+(\d+(?:\.\d+)?)\s*lakhs?', lambda x: float(x)),
            (r'under\s+(\d+(?:\.\d+)?)\s*crores?', lambda x: float(x) * 100),
            (r'below\s+(\d+(?:\.\d+)?)\s*lakhs?', lambda x: float(x)),
            (r'below\s+(\d+(?:\.\d+)?)\s*crores?', lambda x: float(x) * 100),
            # "within X lakhs/crores"
            (r'within\s+(\d+(?:\.\d+)?)\s*lakhs?', lambda x: float(x)),
            (r'within\s+(\d+(?:\.\d+)?)\s*crores?', lambda x: float(x) * 100),
            # "X lakhs budget"
            (r'(\d+(?:\.\d+)?)\s*lakhs?\s*budget', lambda x: float(x)),
            (r'(\d+(?:\.\d+)?)\s*crores?\s*budget', lambda x: float(x) * 100),
            # Range patterns "X-Y lakhs"
            (r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*lakhs?', lambda x, y: (float(x), float(y))),
            (r'(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)\s*lakhs?', lambda x, y: (float(x), float(y))),
        ]
        
        for pattern, converter in price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if len(match.groups()) == 2:  # Range pattern
                    min_price, max_price = converter(match.group(1), match.group(2))
                    constraints['min_price'] = min_price
                    constraints['max_price'] = max_price
                else:  # Single price pattern
                    constraints['max_price'] = converter(match.group(1))
                break
        
        # Budget category detection
        if any(word in query_lower for word in ['budget', 'cheap', 'affordable', 'economical']):
            constraints['budget_category'] = 'budget'
            if not constraints['max_price']:
                constraints['max_price'] = 8  # Default budget limit
        elif any(word in query_lower for word in ['luxury', 'premium', 'high-end', 'expensive']):
            constraints['budget_category'] = 'luxury'
            if not constraints['min_price']:
                constraints['min_price'] = 25  # Default luxury minimum
        elif any(word in query_lower for word in ['mid-range', 'middle', 'moderate']):
            constraints['budget_category'] = 'mid-range'
            if not constraints['min_price']:
                constraints['min_price'] = 8
            if not constraints['max_price']:
                constraints['max_price'] = 25
        
        # Seating capacity
        seating_patterns = [
            (r'(\d+)\s*seater', lambda x: int(x)),
            (r'seats?\s*(\d+)', lambda x: int(x)),
            (r'for\s*(\d+)\s*people', lambda x: int(x))
        ]
        
        for pattern, converter in seating_patterns:
            match = re.search(pattern, query_lower)
            if match:
                constraints['min_seating'] = converter(match.group(1))
                break
        
        # Fuel type
        fuel_types = ['petrol', 'diesel', 'electric', 'hybrid', 'cng', 'lpg']
        for fuel in fuel_types:
            if fuel in query_lower:
                constraints['fuel_type'] = fuel
                break
        
        # Body type
        body_types = ['suv', 'sedan', 'hatchback', 'coupe', 'convertible', 'wagon', 'mpv']
        for body in body_types:
            if body in query_lower:
                constraints['body_type'] = body
                break
        
        # Transmission
        if any(word in query_lower for word in ['automatic', 'auto', 'cvt', 'amt']):
            constraints['transmission'] = 'automatic'
        elif any(word in query_lower for word in ['manual', 'stick']):
            constraints['transmission'] = 'manual'
        
        return constraints
    
    def apply_hard_filters(self, constraints: Dict):
        """Apply hard filters based on constraints"""
        filtered_df = self.df.copy()
        filter_mask = pd.Series([True] * len(filtered_df))
        
        # Price filters
        if constraints['max_price'] is not None:
            price_mask = filtered_df['price_num'] <= constraints['max_price']
            filter_mask &= price_mask
            
        if constraints['min_price'] is not None:
            price_mask = filtered_df['price_num'] >= constraints['min_price']
            filter_mask &= price_mask
        
        # Seating filters
        if constraints['min_seating'] is not None:
            seating_mask = filtered_df['seating_num'] >= constraints['min_seating']
            filter_mask &= seating_mask
        
        # Fuel type filter
        if constraints['fuel_type'] is not None:
            fuel_mask = filtered_df['Fuel_Type'].str.lower().str.contains(
                constraints['fuel_type'], na=False
            )
            filter_mask &= fuel_mask
        
        # Body type filter
        if constraints['body_type'] is not None:
            body_mask = filtered_df['Body_Type'].str.lower().str.contains(
                constraints['body_type'], na=False
            )
            filter_mask &= body_mask
        
        # Transmission filter
        if constraints['transmission'] is not None:
            if constraints['transmission'] == 'automatic':
                trans_mask = filtered_df['Transmission'].str.lower().str.contains(
                    'automatic|cvt|amt', na=False
                )
            else:
                trans_mask = filtered_df['Transmission'].str.lower().str.contains(
                    'manual', na=False
                )
            filter_mask &= trans_mask
        
        filtered_indices = filtered_df.index[filter_mask].tolist()
        
        return filtered_indices

# Initialize recommendation system
recommender = None
try:
    print("Starting to initialize Car recommendation system...")
    recommender = CarRecommendationSystem()
    print("Car recommendation system initialized successfully")
    logger.info("Car recommendation system initialized successfully")
except Exception as e:
    print(f"Failed to initialize recommendation system: {e}")
    logger.error(f"Failed to initialize recommendation system: {e}")
    recommender = None

# Enhanced Pydantic models
class Query(BaseModel):
    query: str = Field(..., description="Natural language query for car recommendation")
    user_preferences: Optional[Dict] = Field(None, description="User preferences for filtering")

class EvalItem(BaseModel):
    query: str
    expected_make: str
    expected_model: str
    expected_variant: str

class EvalRequest(BaseModel):
    items: List[EvalItem]

class RecommendationResponse(BaseModel):
    recommendations: List[Dict]
    query_processed: str
    total_results: int
    processing_time: float

class EvaluationResponse(BaseModel):
    total: int
    correct: int
    accuracy: float
    detailed_results: List[Dict]

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Advanced Car Recommendation System",
        "version": "2.0",
        "model": config.EMBEDDING_MODEL,
        "status": "healthy" if recommender else "initialization_failed",
        "total_cars": len(recommender.df) if recommender else 0,
        "endpoints": {
            "recommend": "/recommend",
            "stats": "/stats",
            "evaluate": "/evaluate"
        },
        "note": "Some endpoints may not work if recommendation system failed to initialize"
    }

@app.get("/test")
async def test():
    """Simple test endpoint that doesn't require recommender system"""
    return {
        "message": "Server is running",
        "timestamp": datetime.now().isoformat(),
        "python_version": "3.10",
        "status": "ok"
    }

def filter_cars(query, df):
    # Hybrid filtering: filter by price, body type, fuel type if mentioned in query
    q = query.lower()
    filtered = df.copy()
    # Price filter (example: 'under 3 lakh', 'below 5 lakh')
    price_match = re.search(r'(under|below|less than) ([0-9]+) ?lakh', q)
    if price_match:
        price_limit = int(price_match.group(2)) * 100000
        def parse_price(p):
            if isinstance(p, str):
                digits = re.sub(r'[^0-9]', '', p)
                return int(digits) if digits else np.inf
            return np.inf
        filtered = filtered[filtered['Price'].apply(parse_price) <= price_limit]
    # Body type filter
    for body in ['hatchback', 'sedan', 'suv', 'mpv', 'convertible', 'coupe', 'wagon']:
        if body in q:
            filtered = filtered[filtered['Body_Type'].str.lower().fillna('').str.contains(body)]
    # Fuel type filter
    for fuel in ['petrol', 'diesel', 'cng', 'electric']:
        if fuel in q:
            filtered = filtered[filtered['Fuel_Type'].str.lower().fillna('').str.contains(fuel)]
    return filtered if not filtered.empty else df

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(query: Query):
    """Get car recommendations based on natural language query"""
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    start_time = datetime.now()
    
    try:
        # Preprocess user query: lowercase and remove special chars
        user_query = query.query.lower()
        user_query = re.sub(r'[^a-z0-9.,\s]', '', user_query)
        user_query = re.sub(r'\s+', ' ', user_query).strip()
        # Hybrid filter
        filtered_df = filter_cars(user_query, recommender.df)
        filtered_descriptions = filtered_df["description"].tolist()
        if not filtered_descriptions:
            return RecommendationResponse(
                recommendations=[],
                query_processed=recommender.clean_text(query.query),
                total_results=0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        # Get indices of filtered cars
        filtered_indices = filtered_df.index.tolist()
        
        if not filtered_indices:
            return RecommendationResponse(
                recommendations=[],
                query_processed=recommender.clean_text(query.query),
                total_results=0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
        filtered_embeddings = recommender.car_embeddings[filtered_indices]
        user_embedding = recommender.model.encode(user_query, convert_to_tensor=True)
        similarity_scores = util.cos_sim(user_embedding, filtered_embeddings)[0]
        
        # Ensure we don't try to get more results than available
        k = min(20, len(filtered_df), len(filtered_indices))
        if k == 0:
            return RecommendationResponse(
                recommendations=[],
                query_processed=recommender.clean_text(query.query),
                total_results=0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
        top_k = torch.topk(similarity_scores, k=k)
        indices = top_k.indices.tolist()
        scores = top_k.values.tolist()
        # Ensure unique models in recommendations
        seen_models = set()
        recommendations = []
        for i, idx in enumerate(indices):
            car = filtered_df.iloc[idx]
            make_model = (str(car.get("Make", "")).strip().lower(), str(car.get("Model", "")).strip().lower())
            if make_model in seen_models:
                continue
            seen_models.add(make_model)
            def safe_str(x):
                if isinstance(x, (np.generic, np.bool_)):
                    return str(x.item())
                return str(x)
            recommendations.append({
                "Make": safe_str(car.get("Make", "")),
                "Model": safe_str(car.get("Model", "")),
                "Variant": safe_str(car.get("Variant", "")),
                "Body_Type": safe_str(car.get("Body_Type", "")),
                "Fuel_Type": safe_str(car.get("Fuel_Type", "")),
                "Power": safe_str(car.get("Power", "")),
                "Price": safe_str(car.get("Price", "")),
                "ARAI_Certified_Mileage": safe_str(car.get("ARAI_Certified_Mileage", "")),
                "Seating_Capacity": safe_str(car.get("Seating_Capacity", "")),
                "description": safe_str(car.get("description", "")),
                "score": float(scores[i])
            })
            if len(recommendations) >= config.TOP_K_RECOMMENDATIONS:
                break
        logger.info(f"Query: {query.query} | Top Recommendation: {recommendations[0] if recommendations else 'None'}")
        return RecommendationResponse(
            recommendations=recommendations,
            query_processed=recommender.clean_text(query.query),
            total_results=len(recommendations),
            processing_time=(datetime.now() - start_time).total_seconds()
        )
    except Exception as e:
        logger.error(f"Error processing recommendation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(eval_request: EvalRequest):
    """Evaluate recommendation system performance"""
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    correct = 0
    total = len(eval_request.items)
    detailed_results = []
    
    for item in eval_request.items:
        try:
            recommendations = recommender.get_recommendations(item.query)
            
            found = False
            found_rank = -1
            
            for i, rec in enumerate(recommendations):
                if (
                    rec["Make"].lower() == item.expected_make.lower() and
                    rec["Model"].lower() == item.expected_model.lower() and
                    rec["Variant"].lower() == item.expected_variant.lower()
                ):
                    found = True
                    found_rank = i + 1
                    break
            
            if found:
                correct += 1
            
            detailed_results.append({
                "query": item.query,
                "expected": f"{item.expected_make} {item.expected_model} {item.expected_variant}",
                "found": found,
                "rank": found_rank,
                "top_result": f"{recommendations[0]['Make']} {recommendations[0]['Model']} {recommendations[0]['Variant']}" if recommendations else "None"
            })
            
        except Exception as e:
            logger.error(f"Error evaluating query '{item.query}': {e}")
            detailed_results.append({
                "query": item.query,
                "expected": f"{item.expected_make} {item.expected_model} {item.expected_variant}",
                "found": False,
                "rank": -1,
                "error": str(e)
            })
    
    accuracy = correct / total if total > 0 else 0
    
    logger.info(f"Evaluation completed: {correct}/{total} correct, accuracy={accuracy:.2f}")
    
    return EvaluationResponse(
        total=total,
        correct=correct,
        accuracy=accuracy,
        detailed_results=detailed_results
    )

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    return {
        "total_cars": len(recommender.df),
        "model_used": config.EMBEDDING_MODEL,
        "embedding_dimensions": recommender.car_embeddings.shape[1] if recommender.car_embeddings is not None else 0,
        "makes_available": sorted(recommender.df['Make'].unique().tolist()),
        "body_types_available": sorted(recommender.df['Body_Type'].unique().tolist()),
        "fuel_types_available": sorted(recommender.df['Fuel_Type'].unique().tolist())
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)