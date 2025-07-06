import requests
import json
import time
from typing import List, Dict

# API endpoint (adjust if your server runs on different host/port)
BASE_URL = "http://localhost:8000"

class CarRecommendationTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.test_results = []
    
    def test_health(self):
        """Test if the API is running"""
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                print("✅ API is running successfully")
                print(f"📊 System info: {response.json()}")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to API: {e}")
            return False
    
    def get_system_stats(self):
        """Get system statistics"""
        try:
            response = requests.get(f"{self.base_url}/stats")
            if response.status_code == 200:
                stats = response.json()
                print("\n📊 System Statistics:")
                print(f"   Total cars: {stats['total_cars']}")
                print(f"   Model used: {stats['model_used']}")
                print(f"   Embedding dimensions: {stats['embedding_dimensions']}")
                print(f"   Available makes: {len(stats['makes_available'])}")
                print(f"   Available body types: {stats['body_types_available']}")
                print(f"   Available fuel types: {stats['fuel_types_available']}")
                return stats
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            return None
    
    def test_single_query(self, query: str, description: str = ""):
        """Test a single query"""
        print(f"\n🔍 Testing: {description if description else query}")
        print(f"   Query: '{query}'")
        
        try:
            payload = {"query": query}
            start_time = time.time()
            
            response = requests.post(f"{self.base_url}/recommend", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                
                print(f"   ✅ Success! Processing time: {processing_time:.3f}s")
                print(f"   📈 Results: {result['total_results']} recommendations")
                print(f"   🎯 Top recommendation: {result['recommendations'][0]['Make']} {result['recommendations'][0]['Model']} {result['recommendations'][0]['Variant']}")
                print(f"   📊 Confidence: {result['recommendations'][0]['match_confidence']} (Score: {result['recommendations'][0]['similarity_score']:.3f})")
                
                # Store result for analysis
                self.test_results.append({
                    "query": query,
                    "description": description,
                    "success": True,
                    "processing_time": processing_time,
                    "total_results": result['total_results'],
                    "top_result": result['recommendations'][0],
                    "all_results": result['recommendations']
                })
                
                return result
            else:
                print(f"   ❌ Failed with status: {response.status_code}")
                print(f"   Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def run_comprehensive_tests(self):
        """Run a comprehensive set of test queries"""
        
        # Test queries covering different scenarios
        test_cases = [
            # Basic brand/model queries
            ("Toyota Camry", "Specific model search"),
            ("Honda Civic", "Popular sedan search"),
            ("BMW X5", "Luxury SUV search"),
            ("Maruti Swift", "Popular hatchback search"),
            
            # Natural language queries
            ("I want a fuel efficient family car", "Natural language - family car"),
            ("Looking for a sporty car under 20 lakhs", "Natural language - sporty + budget"),
            ("Need a diesel SUV for highway driving", "Natural language - diesel SUV"),
            ("Best hybrid car for city driving", "Natural language - hybrid city car"),
            
            # Feature-based queries
            ("automatic transmission sedan", "Feature-based - automatic sedan"),
            ("7 seater SUV", "Feature-based - 7 seater"),
            ("electric car", "Feature-based - electric"),
            ("manual transmission hatchback", "Feature-based - manual hatchback"),
            
            # Price-based queries
            ("budget car under 5 lakhs", "Price range - budget"),
            ("luxury car above 50 lakhs", "Price range - luxury"),
            ("mid range sedan 10-15 lakhs", "Price range - mid-range"),
            ("premium SUV 25-40 lakhs", "Price range - premium SUV"),
            
            # Use case specific
            ("first car for college student", "Use case - college student"),
            ("family car for 6 people", "Use case - large family"),
            ("car for long distance travel", "Use case - long distance"),
            ("city car for daily commute", "Use case - city commute"),
            
            # Performance queries
            ("high performance car", "Performance - high performance"),
            ("most fuel efficient car", "Performance - fuel efficient"),
            ("powerful engine car", "Performance - powerful engine"),
            ("low maintenance car", "Performance - low maintenance"),
            
            # Brand specific queries
            ("German luxury car", "Brand category - German luxury"),
            ("Japanese reliable car", "Brand category - Japanese reliable"),
            ("Korean SUV", "Brand category - Korean SUV"),
            ("Indian brand hatchback", "Brand category - Indian hatchback"),
            
            # Ambiguous/challenging queries
            ("best car", "Ambiguous - best car"),
            ("cheap and good", "Ambiguous - cheap and good"),
            ("something reliable", "Ambiguous - reliable"),
            ("car for new driver", "Ambiguous - new driver"),
            
            # Edge cases
            ("convertible sports car", "Edge case - convertible"),
            ("off road vehicle", "Edge case - off road"),
            ("car with sunroof", "Edge case - specific feature"),
            ("diesel automatic SUV with 7 seats", "Complex - multiple criteria"),
        ]
        
        print("🚀 Starting comprehensive test suite...")
        
        for query, description in test_cases:
            self.test_single_query(query, description)
            time.sleep(0.5)  # Small delay between requests
    
    def run_evaluation_test(self):
        """Run evaluation with sample expected results"""
        print("\n📊 Running evaluation test...")
        
        # Sample evaluation data (adjust based on your actual dataset)
        eval_data = {
            "items": [
                {
                    "query": "Toyota Camry",
                    "expected_make": "Toyota",
                    "expected_model": "Camry",
                    "expected_variant": "Standard"
                },
                {
                    "query": "Honda Civic",
                    "expected_make": "Honda",
                    "expected_model": "Civic",
                    "expected_variant": "Standard"
                },
                {
                    "query": "BMW X5",
                    "expected_make": "BMW",
                    "expected_model": "X5",
                    "expected_variant": "Standard"
                },
                {
                    "query": "fuel efficient family car",
                    "expected_make": "Toyota",
                    "expected_model": "Corolla",
                    "expected_variant": "Standard"
                },
                {
                    "query": "luxury sedan",
                    "expected_make": "Mercedes-Benz",
                    "expected_model": "C-Class",
                    "expected_variant": "Standard"
                }
            ]
        }
        
        try:
            response = requests.post(f"{self.base_url}/evaluate", json=eval_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Evaluation completed!")
                print(f"   📈 Accuracy: {result['accuracy']:.2%}")
                print(f"   📊 Correct: {result['correct']}/{result['total']}")
                
                # Show detailed results
                print("\n📋 Detailed Results:")
                for item in result['detailed_results']:
                    status = "✅" if item['found'] else "❌"
                    print(f"   {status} Query: '{item['query']}'")
                    print(f"      Expected: {item['expected']}")
                    print(f"      Top Result: {item['top_result']}")
                    if item['found']:
                        print(f"      Found at rank: {item['rank']}")
                    print()
                
                return result
            else:
                print(f"❌ Evaluation failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return None
    
    def analyze_results(self):
        """Analyze test results"""
        if not self.test_results:
            print("No test results to analyze")
            return
        
        print("\n📊 TEST RESULTS ANALYSIS")
        print("=" * 50)
        
        successful_tests = [r for r in self.test_results if r['success']]
        
        print(f"✅ Successful tests: {len(successful_tests)}/{len(self.test_results)}")
        
        if successful_tests:
            avg_processing_time = sum(r['processing_time'] for r in successful_tests) / len(successful_tests)
            print(f"⏱️  Average processing time: {avg_processing_time:.3f}s")
            
            # Confidence distribution
            confidence_counts = {}
            for result in successful_tests:
                confidence = result['top_result']['match_confidence']
                confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
            
            print(f"📈 Confidence distribution:")
            for confidence, count in sorted(confidence_counts.items()):
                print(f"   {confidence}: {count} queries")
            
            # Top performing queries
            print(f"\n🏆 Best performing queries:")
            sorted_results = sorted(successful_tests, 
                                  key=lambda x: x['top_result']['similarity_score'], 
                                  reverse=True)
            
            for i, result in enumerate(sorted_results[:5]):
                print(f"   {i+1}. '{result['query']}' - Score: {result['top_result']['similarity_score']:.3f}")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 COMPREHENSIVE CAR RECOMMENDATION SYSTEM TEST")
        print("=" * 60)
        
        # Test API health
        if not self.test_health():
            print("❌ API is not accessible. Please start the server first.")
            return
        
        # Get system stats
        self.get_system_stats()
        
        # Run comprehensive tests
        self.run_comprehensive_tests()
        
        # Run evaluation
        self.run_evaluation_test()
        
        # Analyze results
        self.analyze_results()
        
        print("\n🎉 Testing completed!")

# Quick test functions for individual scenarios
def quick_test_basic():
    """Quick test with basic queries"""
    tester = CarRecommendationTester()
    queries = [
        "Toyota Camry",
        "fuel efficient car",
        "luxury SUV",
        "budget hatchback"
    ]
    
    for query in queries:
        tester.test_single_query(query)

def quick_test_natural_language():
    """Test natural language queries"""
    tester = CarRecommendationTester()
    queries = [
        "I need a reliable family car for 5 people",
        "Looking for a sporty car under 25 lakhs",
        "Best car for daily office commute in city",
        "Fuel efficient automatic car for highway driving"
    ]
    
    for query in queries:
        tester.test_single_query(query)

def quick_test_edge_cases():
    """Test edge cases and challenging queries"""
    tester = CarRecommendationTester()
    queries = [
        "best car",
        "something good and cheap",
        "car for new driver",
        "diesel automatic 7 seater SUV under 30 lakhs"
    ]
    
    for query in queries:
        tester.test_single_query(query)

if __name__ == "__main__":
    # Run comprehensive test suite
    tester = CarRecommendationTester()
    tester.run_all_tests()
    
    # Uncomment to run specific quick tests:
    # quick_test_basic()
    # quick_test_natural_language()
    # quick_test_edge_cases()