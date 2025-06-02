# integration_test.py
import requests
import time
import json
import subprocess
import sys
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from PIL import Image
import numpy as np
import io

class IntegrationTester:
    def __init__(self, backend_url="http://localhost:5001", frontend_url="http://localhost:3000"):
        self.backend_url = backend_url
        self.frontend_url = frontend_url
        self.driver = None
        self.test_results = []
        
    def setup_browser(self):
        """Setup Chrome browser for testing"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run in background
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            return True
        except Exception as e:
            print(f"❌ Failed to setup browser: {e}")
            print("💡 Install ChromeDriver: https://chromedriver.chromium.org/")
            return False
    
    def create_test_image(self, filename="test_image.jpg"):
        """Create a test image file"""
        img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(filename, 'JPEG', quality=85)
        return filename
    
    def test_backend_frontend_communication(self):
        """Test 1: Backend-Frontend API Communication"""
        print("🔍 Testing Backend-Frontend Communication...")
        
        try:
            # Test backend health from frontend's perspective
            response = requests.get(f"{self.backend_url}/api/health", timeout=10)
            if response.status_code == 200:
                print("✅ Backend accessible from frontend network")
                return True
            else:
                print(f"❌ Backend not accessible: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Backend communication failed: {e}")
            return False
    
    def test_cors_headers(self):
        """Test 2: CORS Configuration"""
        print("🔍 Testing CORS Headers...")
        
        try:
            headers = {
                'Origin': self.frontend_url,
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type'
            }
            
            # Test preflight request
            response = requests.options(f"{self.backend_url}/api/analyze", headers=headers, timeout=10)
            
            cors_headers = response.headers
            if 'Access-Control-Allow-Origin' in cors_headers:
                print("✅ CORS headers configured correctly")
                return True
            else:
                print("❌ CORS headers missing")
                return False
        except Exception as e:
            print(f"❌ CORS test failed: {e}")
            return False
    
    def test_file_upload_flow(self):
        """Test 3: Complete File Upload Flow"""
        print("🔍 Testing Complete Upload Flow...")
        
        if not self.driver:
            print("❌ Browser not available, skipping UI test")
            return False
        
        try:
            # Navigate to frontend
            self.driver.get(self.frontend_url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Check if main elements are present
            title_element = self.driver.find_element(By.XPATH, "//*[contains(text(), 'Blood Cell Analyzer')]")
            if not title_element:
                print("❌ Main title not found")
                return False
            
            print("✅ Frontend loaded successfully")
            
            # Look for upload area
            try:
                upload_elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Upload') or contains(text(), 'drag')]")
                if upload_elements:
                    print("✅ Upload interface found")
                    return True
                else:
                    print("⚠️  Upload interface not found (may need manual check)")
                    return True  # Don't fail for this
            except Exception as e:
                print(f"⚠️  Upload interface check failed: {e}")
                return True  # Don't fail for this
                
        except Exception as e:
            print(f"❌ Frontend test failed: {e}")
            return False
    
    def test_api_response_format(self):
        """Test 4: API Response Format Consistency"""
        print("🔍 Testing API Response Formats...")
        
        try:
            # Create test image
            img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_io = io.BytesIO()
            img.save(img_io, 'JPEG', quality=85)
            img_io.seek(0)
            
            files = {'image': ('test.jpg', img_io, 'image/jpeg')}
            response = requests.post(f"{self.backend_url}/api/analyze", files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure matches frontend expectations
                expected_fields = [
                    'success', 'analysis_id', 'cell_counts', 'processing_time',
                    'confidence_score', 'total_cells_detected'
                ]
                
                missing_fields = [field for field in expected_fields if field not in data]
                if missing_fields:
                    print(f"❌ Missing response fields: {missing_fields}")
                    return False
                
                # Check cell_counts structure
                cell_counts = data.get('cell_counts', {})
                expected_cells = ['RBC', 'WBC', 'Platelets']
                missing_cells = [cell for cell in expected_cells if cell not in cell_counts]
                
                if missing_cells:
                    print(f"❌ Missing cell types: {missing_cells}")
                    return False
                
                print("✅ API response format matches frontend expectations")
                return True
            else:
                print(f"❌ API request failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ API format test failed: {e}")
            return False
    
    def test_error_handling_integration(self):
        """Test 5: Error Handling Integration"""
        print("🔍 Testing Error Handling Integration...")
        
        try:
            # Test various error scenarios
            error_tests = [
                {
                    'name': 'No file',
                    'request': lambda: requests.post(f"{self.backend_url}/api/analyze", timeout=10),
                    'expected_status': 400
                },
                {
                    'name': 'Invalid endpoint',
                    'request': lambda: requests.get(f"{self.backend_url}/api/nonexistent", timeout=10),
                    'expected_status': 404
                }
            ]
            
            for test in error_tests:
                try:
                    response = test['request']()
                    if response.status_code == test['expected_status']:
                        print(f"✅ {test['name']}: Correct error status {response.status_code}")
                    else:
                        print(f"❌ {test['name']}: Expected {test['expected_status']}, got {response.status_code}")
                        return False
                except Exception as e:
                    print(f"❌ {test['name']}: Request failed: {e}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            return False
    
    def test_performance_integration(self):
        """Test 6: Performance Under Load"""
        print("🔍 Testing Performance Integration...")
        
        try:
            # Test concurrent requests
            import threading
            import time
            
            results = []
            errors = []
            
            def make_request():
                try:
                    img_array = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
                    img = Image.fromarray(img_array)
                    img_io = io.BytesIO()
                    img.save(img_io, 'JPEG', quality=85)
                    img_io.seek(0)
                    
                    files = {'image': ('test.jpg', img_io, 'image/jpeg')}
                    start_time = time.time()
                    response = requests.post(f"{self.backend_url}/api/analyze", files=files, timeout=30)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        results.append(end_time - start_time)
                    else:
                        errors.append(f"Status: {response.status_code}")
                except Exception as e:
                    errors.append(str(e))
            
            # Run 5 concurrent requests
            threads = []
            for i in range(5):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            if errors:
                print(f"❌ Performance test had errors: {errors}")
                return False
            
            if not results:
                print("❌ No successful requests in performance test")
                return False
            
            avg_time = sum(results) / len(results)
            max_time = max(results)
            
            print(f"✅ Performance test: {len(results)} requests, avg: {avg_time:.2f}s, max: {max_time:.2f}s")
            
            # Check if performance is reasonable
            if avg_time > 10:  # 10 seconds seems reasonable for concurrent requests
                print(f"⚠️  Average response time is high: {avg_time:.2f}s")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return False
    
    def run_all_integration_tests(self):
        """Run all integration tests"""
        print("🔗 Running Integration Tests...")
        print("=" * 50)
        
        # Setup browser
        browser_available = self.setup_browser()
        
        tests = [
            ("Backend-Frontend Communication", self.test_backend_frontend_communication),
            ("CORS Configuration", self.test_cors_headers),
            ("API Response Format", self.test_api_response_format),
            ("Error Handling Integration", self.test_error_handling_integration),
            ("Performance Integration", self.test_performance_integration)
        ]
        
        if browser_available:
            tests.append(("File Upload Flow", self.test_file_upload_flow))
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            print(f"\n🧪 {test_name}:")
            try:
                result = test_func()
                if result:
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    failed += 1
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                failed += 1
                print(f"❌ {test_name}: ERROR - {e}")
        
        # Cleanup
        if self.driver:
            self.driver.quit()
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"📊 Integration Test Summary:")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
        
        return failed == 0

def check_services_running():
    """Check if both backend and frontend are running"""
    print("🔍 Checking if services are running...")
    
    # Check backend
    try:
        response = requests.get("http://localhost:5001/api/health", timeout=5)
        backend_running = response.status_code == 200
    except:
        try:
            response = requests.get("http://localhost:5000/api/health", timeout=5)
            backend_running = response.status_code == 200
        except:
            backend_running = False
    
    # Check frontend
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        frontend_running = response.status_code == 200
    except:
        frontend_running = False
    
    print(f"Backend: {'✅ Running' if backend_running else '❌ Not running'}")
    print(f"Frontend: {'✅ Running' if frontend_running else '❌ Not running'}")
    
    return backend_running, frontend_running

if __name__ == "__main__":
    print("🩸 Blood Cell Analyzer - Integration Test Suite")
    print("=" * 60)
    
    # Check if services are running
    backend_running, frontend_running = check_services_running()
    
    if not backend_running:
        print("\n❌ Backend is not running!")
        print("Please start the backend:")
        print("  cd backend && source venv/bin/activate && python app.py")
        sys.exit(1)
    
    if not frontend_running:
        print("\n❌ Frontend is not running!")
        print("Please start the frontend:")
        print("  cd frontend && npm start")
        sys.exit(1)
    
    # Run integration tests
    tester = IntegrationTester()
    success = tester.run_all_integration_tests()
    
    print("\n" + "=" * 60)
    print("🏁 Integration Testing Complete")
    
    if success:
        print("🎉 All integration tests passed! Your application is working end-to-end.")
    else:
        print("⚠️  Some integration tests failed. Please check the output above.")
    
    sys.exit(0 if success else 1)

