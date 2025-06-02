
import React, { useState, useEffect } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import '../styles/Dashboard.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalAnalyses: 0,
    averageProcessingTime: 0,
    averageAccuracy: 0,
    recentAnalyses: []
  });
  const [timeRange, setTimeRange] = useState('7d');

  useEffect(() => {
    // Simulate fetching dashboard data
    // In a real app, this would fetch from your backend
    const mockStats = {
      totalAnalyses: 156,
      averageProcessingTime: 1.8,
      averageAccuracy: 93.2,
      recentAnalyses: generateMockAnalyses()
    };
    setStats(mockStats);
  }, [timeRange]);

  const generateMockAnalyses = () => {
    const analyses = [];
    const cellTypes = ['RBC', 'WBC', 'Platelet'];
    
    for (let i = 0; i < 10; i++) {
      analyses.push({
        id: `analysis_${i + 1}`,
        timestamp: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString(),
        cellCounts: {
          RBC: Math.floor(Math.random() * 300) + 200,
          WBC: Math.floor(Math.random() * 50) + 10,
          Platelet: Math.floor(Math.random() * 100) + 20
        },
        processingTime: (Math.random() * 2 + 0.5).toFixed(2),
        accuracy: (Math.random() * 5 + 90).toFixed(1)
      });
    }
    return analyses;
  };

  // Chart data for analysis trends
  const trendData = {
    labels: stats.recentAnalyses.map(analysis => 
      new Date(analysis.timestamp).toLocaleDateString()
    ).reverse(),
    datasets: [
      {
        label: 'Total Cells Detected',
        data: stats.recentAnalyses.map(analysis => 
          Object.values(analysis.cellCounts).reduce((sum, count) => sum + count, 0)
        ).reverse(),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1,
      },
    ],
  };

  // Performance metrics chart
  const performanceData = {
    labels: stats.recentAnalyses.map(analysis => 
      new Date(analysis.timestamp).toLocaleDateString()
    ).reverse(),
    datasets: [
      {
        label: 'Processing Time (s)',
        data: stats.recentAnalyses.map(analysis => parseFloat(analysis.processingTime)).reverse(),
        backgroundColor: 'rgba(255, 99, 132, 0.8)',
        yAxisID: 'y',
      },
      {
        label: 'Accuracy (%)',
        data: stats.recentAnalyses.map(analysis => parseFloat(analysis.accuracy)).reverse(),
        backgroundColor: 'rgba(54, 162, 235, 0.8)',
        yAxisID: 'y1',
      },
    ],
  };

  const performanceOptions = {
    responsive: true,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Date',
        },
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Processing Time (seconds)',
        },
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Accuracy (%)',
        },
        grid: {
          drawOnChartArea: false,
        },
      },
    },
  };

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>📊 Analytics Dashboard</h2>
        <div className="time-range-selector">
          <select 
            value={timeRange} 
            onChange={(e) => setTimeRange(e.target.value)}
            className="time-select"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
          </select>
        </div>
      </div>

      <div className="dashboard-stats">
        <div className="stat-card">
          <div className="stat-icon">🔬</div>
          <div className="stat-content">
            <div className="stat-value">{stats.totalAnalyses}</div>
            <div className="stat-label">Total Analyses</div>
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon">⚡</div>
          <div className="stat-content">
            <div className="stat-value">{stats.averageProcessingTime}s</div>
            <div className="stat-label">Avg Processing Time</div>
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon">🎯</div>
          <div className="stat-content">
            <div className="stat-value">{stats.averageAccuracy}%</div>
            <div className="stat-label">Average Accuracy</div>
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon">📈</div>
          <div className="stat-content">
            <div className="stat-value">
              {stats.recentAnalyses.length > 1 ? '+12%' : '0%'}
            </div>
            <div className="stat-label">Growth Rate</div>
          </div>
        </div>
      </div>

      <div className="dashboard-charts">
        <div className="chart-container">
          <h3>Analysis Trends</h3>
          <Line data={trendData} options={{ responsive: true }} />
        </div>
        
        <div className="chart-container">
          <h3>Performance Metrics</h3>
          <Bar data={performanceData} options={performanceOptions} />
        </div>
      </div>

      <div className="recent-analyses">
        <h3>Recent Analyses</h3>
        <div className="analyses-table">
          <table>
            <thead>
              <tr>
                <th>Analysis ID</th>
                <th>Date</th>
                <th>RBC</th>
                <th>WBC</th>
                <th>Platelets</th>
                <th>Processing Time</th>
                <th>Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {stats.recentAnalyses.slice(0, 5).map((analysis) => (
                <tr key={analysis.id}>
                  <td>{analysis.id}</td>
                  <td>{new Date(analysis.timestamp).toLocaleDateString()}</td>
                  <td>{analysis.cellCounts.RBC}</td>
                  <td>{analysis.cellCounts.WBC}</td>
                  <td>{analysis.cellCounts.Platelet}</td>
                  <td>{analysis.processingTime}s</td>
                  <td>{analysis.accuracy}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="system-status">
        <h3>System Status</h3>
        <div className="status-indicators">
          <div className="status-item">
            <div className="status-dot green"></div>
            <span>API Server: Online</span>
          </div>
          <div className="status-item">
            <div className="status-dot green"></div>
            <span>ML Model: Ready</span>
          </div>
          <div className="status-item">
            <div className="status-dot green"></div>
            <span>Database: Connected</span>
          </div>
          <div className="status-item">
            <div className="status-dot yellow"></div>
            <span>Firebase: Syncing</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;