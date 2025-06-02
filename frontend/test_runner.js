// frontend/test_runner.js
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class FrontendTester {
  constructor() {
    this.testResults = [];
    this.passed = 0;
    this.failed = 0;
  }

  logResult(testName, passed, message = '') {
    const status = passed ? '✅ PASS' : '❌ FAIL';
    console.log(`${status}: ${testName}`);
    if (message) {
      console.log(`   ${message}`);
    }

    this.testResults.push({
      test: testName,
      passed,
      message
    });

    if (passed) {
      this.passed++;
    } else {
      this.failed++;
    }
  }

  checkDependencies() {
    console.log('🔍 Checking Frontend Dependencies...');
    
    try {
      const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      const nodeModulesExists = fs.existsSync('node_modules');
      
      if (!nodeModulesExists) {
        this.logResult('Dependencies Check', false, 'node_modules folder not found. Run: npm install');
        return false;
      }

      // Check critical dependencies
      const criticalDeps = ['react', 'react-dom'];
      const missingDeps = criticalDeps.filter(dep => 
        !packageJson.dependencies[dep] && !packageJson.devDependencies[dep]
      );

      if (missingDeps.length > 0) {
        this.logResult('Dependencies Check', false, `Missing dependencies: ${missingDeps.join(', ')}`);
        return false;
      }

      this.logResult('Dependencies Check', true, `${Object.keys(packageJson.dependencies).length} dependencies found`);
      return true;
    } catch (error) {
      this.logResult('Dependencies Check', false, `Error: ${error.message}`);
      return false;
    }
  }

  checkFileStructure() {
    console.log('🗂️  Checking File Structure...');
    
    const requiredFiles = [
      'src/index.js',
      'src/App.js',
      'public/index.html',
      'package.json'
    ];

    const missingFiles = requiredFiles.filter(file => !fs.existsSync(file));
    
    if (missingFiles.length > 0) {
      this.logResult('File Structure', false, `Missing files: ${missingFiles.join(', ')}`);
      return false;
    }

    this.logResult('File Structure', true, 'All required files present');
    return true;
  }

  runJestTests() {
    console.log('🧪 Running Jest Tests...');
    
    try {
      // Check if any test files exist
      const testFiles = [];
      const searchDirs = ['src/__tests__', 'src'];
      
      for (const dir of searchDirs) {
        if (fs.existsSync(dir)) {
          const files = fs.readdirSync(dir, { recursive: true });
          testFiles.push(...files.filter(f => f.includes('.test.') || f.includes('.spec.')));
        }
      }

      if (testFiles.length === 0) {
        this.logResult('Unit Tests (Jest)', true, 'No test files found, skipping...');
        return true;
      }

      const output = execSync('npm test -- --coverage --watchAll=false --ci', {
        encoding: 'utf8',
        stdio: 'pipe',
        timeout: 120000 // 2 minutes
      });

      this.logResult('Unit Tests (Jest)', true, 'Tests completed successfully');
      return true;
    } catch (error) {
      const errorOutput = error.stdout || error.stderr || error.message;
      
      if (errorOutput.includes('No tests found')) {
        this.logResult('Unit Tests (Jest)', true, 'No test files found, skipping...');
        return true;
      }
      
      this.logResult('Unit Tests (Jest)', false, 'Tests failed to run');
      return false;
    }
  }

  checkBuild() {
    console.log('🏗️  Testing Production Build...');
    
    try {
      // Clean previous build
      if (fs.existsSync('build')) {
        fs.rmSync('build', { recursive: true, force: true });
      }

      const output = execSync('npm run build', {
        encoding: 'utf8',
        stdio: 'pipe',
        timeout: 300000 // 5 minutes
      });

      // Check if build folder was created
      if (!fs.existsSync('build')) {
        this.logResult('Production Build', false, 'Build folder not created');
        return false;
      }

      this.logResult('Production Build', true, 'Build completed successfully');
      return true;
    } catch (error) {
      this.logResult('Production Build', false, `Build failed: ${error.message}`);
      return false;
    }
  }

  checkEnvironmentConfig() {
    console.log('⚙️  Checking Environment Configuration...');
    
    try {
      const envExists = fs.existsSync('.env');
      const envLocalExists = fs.existsSync('.env.local');
      
      if (!envExists && !envLocalExists) {
        // Create a basic .env file
        fs.writeFileSync('.env', 'REACT_APP_API_URL=http://localhost:5001\nREACT_APP_VERSION=1.0.0\n');
        this.logResult('Environment Config', true, 'Created basic .env file');
        return true;
      }

      this.logResult('Environment Config', true, 'Environment configuration found');
      return true;
    } catch (error) {
      this.logResult('Environment Config', false, `Error: ${error.message}`);
      return false;
    }
  }

  runAllTests() {
    console.log('🎨 Running Frontend Test Suite...');
    console.log('='.repeat(50));

    const tests = [
      this.checkDependencies.bind(this),
      this.checkFileStructure.bind(this),
      this.checkEnvironmentConfig.bind(this),
      this.runJestTests.bind(this),
      this.checkBuild.bind(this)
    ];

    for (const test of tests) {
      test();
      console.log(''); // Add spacing between tests
    }

    // Print summary
    console.log('='.repeat(50));
    console.log('📊 Frontend Test Summary:');
    console.log(`✅ Passed: ${this.passed}`);
    console.log(`❌ Failed: ${this.failed}`);
    console.log(`📈 Success Rate: ${((this.passed / (this.passed + this.failed)) * 100).toFixed(1)}%`);

    return this.failed === 0;
  }
}

// Main execution
if (require.main === module) {
  console.log('🩸 Blood Cell Analyzer - Frontend Test Suite');
  console.log('='.repeat(60));

  // Ensure we're in the frontend directory
  if (!fs.existsSync('package.json')) {
    console.log('❌ Please run from the frontend directory');
    process.exit(1);
  }

  const tester = new FrontendTester();
  const success = tester.runAllTests();

  console.log('\n' + '='.repeat(60));
  console.log('🏁 Frontend Testing Complete');
  
  if (success) {
    console.log('🎉 All frontend tests passed!');
    process.exit(0);
  } else {
    console.log('⚠️  Some tests failed. Please check the output above.');
    process.exit(1);
  }
}

module.exports = FrontendTester;
