// frontend/src/index.js
import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { ErrorBoundary } from 'react-error-boundary';
import App from './App';
import ErrorFallback from './components/ErrorFallback';
import { AnalyticsProvider } from './contexts/AnalyticsContext';
import { NotificationProvider } from './contexts/NotificationContext';
import { ConfigProvider } from './contexts/ConfigContext';
import { ThemeProvider } from './contexts/ThemeContext';
import reportWebVitals from './utils/reportWebVitals';
import { initializeApp } from './utils/appInitialization';
import './index.css';

// Import Firebase configuration
import './firebase';

// Performance monitoring
const startTime = performance.now();

/**
 * Error boundary error handler
 */
function handleError(error, errorInfo) {
  console.error('React Error Boundary caught an error:', error, errorInfo);
  
  // Log to analytics service if available
  if (window.gtag) {
    window.gtag('event', 'exception', {
      description: error.message,
      fatal: true,
      error_boundary: true
    });
  }
  
  // Log to external error reporting service (e.g., Sentry)
  if (window.Sentry) {
    window.Sentry.captureException(error, {
      contexts: {
        react: {
          componentStack: errorInfo.componentStack
        }
      }
    });
  }
  
  // Log error details for debugging
  const errorDetails = {
    message: error.message,
    stack: error.stack,
    componentStack: errorInfo.componentStack,
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    url: window.location.href
  };
  
  // Store error in localStorage for potential recovery
  try {
    localStorage.setItem('lastError', JSON.stringify(errorDetails));
  } catch (e) {
    console.warn('Could not store error details in localStorage:', e);
  }
}

/**
 * App initialization and mounting
 */
async function initializeAndMountApp() {
  try {
    // Get the root container
    const container = document.getElementById('root');
    if (!container) {
      throw new Error('Root container not found');
    }
    
    // Initialize the React 18 root
    const root = createRoot(container);
    
    // Initialize app services
    console.log('🚀 Initializing Blood Cell Analyzer...');
    await initializeApp();
    
    // Calculate initialization time
    const initTime = performance.now() - startTime;
    console.log(`✅ App initialized in ${initTime.toFixed(2)}ms`);
    
    // Performance mark
    performance.mark('app-mount-start');
    
    // Render the app with all providers
    root.render(
      <React.StrictMode>
        <ErrorBoundary
          FallbackComponent={ErrorFallback}
          onError={handleError}
          onReset={() => {
            // Clear any error state and reload
            localStorage.removeItem('lastError');
            window.location.reload();
          }}
        >
          <BrowserRouter>
            <ConfigProvider>
              <ThemeProvider>
                <AnalyticsProvider>
                  <NotificationProvider>
                    <App />
                  </NotificationProvider>
                </AnalyticsProvider>
              </ThemeProvider>
            </ConfigProvider>
          </BrowserRouter>
        </ErrorBoundary>
      </React.StrictMode>
    );
    
    // Performance mark
    performance.mark('app-mount-end');
    performance.measure('app-mount', 'app-mount-start', 'app-mount-end');
    
    // Hide loading screen
    hideLoadingScreen();
    
    // Initialize performance monitoring
    initializePerformanceMonitoring();
    
    console.log('🩸 Blood Cell Analyzer loaded successfully!');
    
  } catch (error) {
    console.error('Failed to initialize app:', error);
    showFatalError(error);
  }
}

/**
 * Hide the loading screen with smooth transition
 */
function hideLoadingScreen() {
  const loadingScreen = document.getElementById('app-loading');
  if (loadingScreen) {
    // Add a small delay to ensure React has rendered
    setTimeout(() => {
      loadingScreen.classList.add('hidden');
      
      // Remove from DOM after transition
      setTimeout(() => {
        if (loadingScreen.parentNode) {
          loadingScreen.parentNode.removeChild(loadingScreen);
        }
      }, 500);
    }, 100);
  }
}

/**
 * Show fatal error when app fails to initialize
 */
function showFatalError(error) {
  const container = document.getElementById('root');
  const loadingScreen = document.getElementById('app-loading');
  
  // Hide loading screen
  if (loadingScreen) {
    loadingScreen.style.display = 'none';
  }
  
  // Show error message
  if (container) {
    container.innerHTML = `
      <div style="
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      ">
        <div style="
          background: rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(10px);
          padding: 3rem;
          border-radius: 20px;
          max-width: 500px;
          width: 100%;
        ">
          <div style="font-size: 4rem; margin-bottom: 1rem;">💥</div>
          <h1 style="font-size: 1.5rem; margin-bottom: 1rem; font-weight: 600;">
            Application Failed to Load
          </h1>
          <p style="margin-bottom: 2rem; opacity: 0.9; line-height: 1.6;">
            We're sorry, but the Blood Cell Analyzer failed to initialize. 
            This could be due to a network issue or browser compatibility problem.
          </p>
          <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
            <button 
              onclick="window.location.reload()" 
              style="
                background: rgba(255, 255, 255, 0.2);
                color: white;
                border: 2px solid white;
                padding: 0.75rem 1.5rem;
                border-radius: 25px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
              "
              onmouseover="this.style.background='rgba(255,255,255,0.3)'"
              onmouseout="this.style.background='rgba(255,255,255,0.2)'"
            >
              Reload Page
            </button>
            <a 
              href="mailto:support@blood-cell-analyzer.com"
              style="
                background: white;
                color: #667eea;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 25px;
                text-decoration: none;
                font-weight: 500;
                transition: all 0.3s ease;
                display: inline-block;
              "
            >
              Contact Support
            </a>
          </div>
          <details style="margin-top: 2rem; text-align: left;">
            <summary style="cursor: pointer; margin-bottom: 1rem;">
              Technical Details
            </summary>
            <pre style="
              background: rgba(0, 0, 0, 0.2);
              padding: 1rem;
              border-radius: 8px;
              overflow: auto;
              font-size: 0.8rem;
              white-space: pre-wrap;
            ">${error.message}\n\n${error.stack}</pre>
          </details>
        </div>
      </div>
    `;
  }
}

/**
 * Initialize performance monitoring
 */
function initializePerformanceMonitoring() {
  // Report Web Vitals
  reportWebVitals((metric) => {
    console.log('📊 Web Vital:', metric);
    
    // Send to analytics if available
    if (window.gtag) {
      window.gtag('event', metric.name, {
        event_category: 'Web Vitals',
        event_label: metric.id,
        value: Math.round(metric.name === 'CLS' ? metric.value * 1000 : metric.value),
        non_interaction: true,
      });
    }
  });
  
  // Log performance metrics
  setTimeout(() => {
    const navigation = performance.getEntriesByType('navigation')[0];
    const paint = performance.getEntriesByType('paint');
    
    console.log('🔍 Performance Metrics:', {
      domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
      loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
      firstPaint: paint.find(p => p.name === 'first-paint')?.startTime,
      firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime,
      totalLoadTime: performance.now() - startTime
    });
  }, 1000);
}

/**
 * Handle browser compatibility checks
 */
function checkBrowserCompatibility() {
  const incompatibilityReasons = [];
  
  // Check for essential features
  if (!window.Promise) {
    incompatibilityReasons.push('Promises not supported');
  }
  
  if (!window.fetch) {
    incompatibilityReasons.push('Fetch API not supported');
  }
  
  if (!window.FileReader) {
    incompatibilityReasons.push('File API not supported');
  }
  
  if (!Array.prototype.includes) {
    incompatibilityReasons.push('Modern Array methods not supported');
  }
  
  // Check for modern JavaScript features
  try {
    // Test arrow functions and const/let
    eval('const test = () => {}');
  } catch (e) {
    incompatibilityReasons.push('Modern JavaScript syntax not supported');
  }
  
  if (incompatibilityReasons.length > 0) {
    console.warn('⚠️ Browser compatibility issues:', incompatibilityReasons);
    
    // Show compatibility warning
    const warning = document.createElement('div');
    warning.innerHTML = `
      <div style="
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: #ff9500;
        color: white;
        padding: 1rem;
        text-align: center;
        z-index: 10000;
        font-family: system-ui, sans-serif;
      ">
        ⚠️ Your browser may not be fully compatible with this application. 
        Please consider updating to a modern browser for the best experience.
        <button 
          onclick="this.parentNode.parentNode.removeChild(this.parentNode)"
          style="
            background: none;
            border: 1px solid white;
            color: white;
            padding: 0.25rem 0.5rem;
            margin-left: 1rem;
            border-radius: 4px;
            cursor: pointer;
          "
        >
          Dismiss
        </button>
      </div>
    `;
    document.body.appendChild(warning);
  }
}

/**
 * Handle offline/online status
 */
function setupNetworkStatusHandling() {
  function updateNetworkStatus() {
    const isOnline = navigator.onLine;
    console.log(`🌐 Network status: ${isOnline ? 'Online' : 'Offline'}`);
    
    // Dispatch custom event for app to handle
    window.dispatchEvent(new CustomEvent('networkStatusChange', {
      detail: { isOnline }
    }));
    
    // Show/hide offline indicator
    let offlineIndicator = document.getElementById('offline-indicator');
    
    if (!isOnline && !offlineIndicator) {
      offlineIndicator = document.createElement('div');
      offlineIndicator.id = 'offline-indicator';
      offlineIndicator.innerHTML = `
        <div style="
          position: fixed;
          bottom: 1rem;
          left: 1rem;
          background: #ff4757;
          color: white;
          padding: 0.75rem 1rem;
          border-radius: 25px;
          font-size: 0.9rem;
          z-index: 9999;
          box-shadow: 0 4px 12px rgba(255, 71, 87, 0.3);
          font-family: system-ui, sans-serif;
        ">
          📡 You're offline
        </div>
      `;
      document.body.appendChild(offlineIndicator);
    } else if (isOnline && offlineIndicator) {
      offlineIndicator.remove();
    }
  }
  
  window.addEventListener('online', updateNetworkStatus);
  window.addEventListener('offline', updateNetworkStatus);
  updateNetworkStatus(); // Initial check
}

/**
 * Setup development helpers
 */
function setupDevelopmentHelpers() {
  if (process.env.NODE_ENV === 'development') {
    // Add development helpers to window for debugging
    window.bloodCellAnalyzer = {
      clearStorage: () => {
        localStorage.clear();
        sessionStorage.clear();
        console.log('🧹 Storage cleared');
      },
      getPerformance: () => {
        return {
          navigation: performance.getEntriesByType('navigation')[0],
          paint: performance.getEntriesByType('paint'),
          measure: performance.getEntriesByType('measure'),
          mark: performance.getEntriesByType('mark')
        };
      },
      getFeatures: () => window.FEATURES,
      getConfig: () => process.env
    };
    
    console.log('🛠️ Development helpers available at window.bloodCellAnalyzer');
  }
}

/**
 * Main initialization function
 */
function main() {
  console.log('🩸 Blood Cell Analyzer v' + (process.env.REACT_APP_VERSION || '1.0.0'));
  console.log('🔧 Build:', process.env.NODE_ENV);
  
  // Check browser compatibility
  checkBrowserCompatibility();
  
  // Setup network status handling
  setupNetworkStatusHandling();
  
  // Setup development helpers
  setupDevelopmentHelpers();
  
  // Initialize and mount the app
  initializeAndMountApp();
}

/**
 * Wait for DOM to be ready, then initialize
 */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', main);
} else {
  main();
}

/**
 * Handle page visibility changes for performance optimization
 */
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    console.log('📱 App went to background');
    // Pause non-critical operations
  } else {
    console.log('📱 App came to foreground');
    // Resume operations
  }
});

/**
 * Handle beforeunload for cleanup
 */
window.addEventListener('beforeunload', (event) => {
  // Perform cleanup if needed
  console.log('👋 App is unloading');
  
  // Cancel any pending requests
  if (window.abortController) {
    window.abortController.abort();
  }
});

/**
 * Export for testing
 */
if (process.env.NODE_ENV === 'test') {
  window.__testHelpers = {
    initializeAndMountApp,
    hideLoadingScreen,
    showFatalError,
    handleError
  };
}