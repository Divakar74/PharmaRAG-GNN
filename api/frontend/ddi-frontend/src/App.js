import React, { useState, useEffect } from 'react';
import axios from 'axios';
import AsyncSelect from 'react-select/async';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import ForceGraph2D from 'react-force-graph-2d';

function DrugCard({ drug }) {
    if (!drug) return null;
    return (
      <div className="mt-2 p-4 bg-gray-50 border border-gray-200 rounded-lg shadow-sm">
        <h3 className="text-lg font-semibold text-gray-800">{drug.label}</h3>
        <p className="text-sm text-gray-600">Value: {drug.value}</p>
      </div>
    );
  }

function App() {
  const [drugA, setDrugA] = useState(null);
  const [drugB, setDrugB] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [evidence, setEvidence] = useState(null);
  const [aiSummary, setAiSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [userId, setUserId] = useState('');
  const [history, setHistory] = useState([]);

  useEffect(() => {
    let storedUserId = localStorage.getItem('userId');
    if (!storedUserId) {
      storedUserId = 'user_' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('userId', storedUserId);
    }
    setUserId(storedUserId);
  }, []);

  const loadOptions = async (inputValue) => {
    if (!inputValue) return [];
    try {
      const response = await axios.get('/search_drug', { params: { q: inputValue, limit: 10 } });
      return response.data.results.map(drug => ({ value: drug.name || drug.id, label: drug.name || drug.id }));
    } catch (err) {
      return [];
    }
  };

  const handlePredict = async () => {
    if (!drugA || !drugB) {
      setError('Please select both drugs');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('/predict_ddi', {
        drug_a: drugA.value,
        drug_b: drugB.value,
        user_id: userId
      });
      setPrediction(response.data);
      setGraphData({
        nodes: [{ id: drugA.value, type: 'drug' }, { id: drugB.value, type: 'drug' }],
        links: response.data.interaction_probability > 0.5 ? [{ source: drugA.value, target: drugB.value }] : []
      });
    } catch (err) {
      setError(err.response?.data?.error || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleHistory = async () => {
    try {
      const response = await axios.get('/history', { params: { limit: 10, user_id: userId } });
      setHistory(response.data.history);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">Drug-Drug Interaction Predictor</h1>
          <div className="text-sm text-gray-600">Session ID: {userId}</div>
        </header>

        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold text-gray-800 mb-6">Select Drugs</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Drug A</label>
              <AsyncSelect loadOptions={loadOptions} onChange={setDrugA} placeholder="Search for drug A" />
              <DrugCard drug={drugA} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Drug B</label>
              <AsyncSelect loadOptions={loadOptions} onChange={setDrugB} placeholder="Search for drug B" />
              <DrugCard drug={drugB} />
            </div>
          </div>
          <button onClick={handlePredict} disabled={loading}
            className="w-full bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-blue-700 disabled:opacity-50">
            {loading ? 'Analyzing...' : 'Predict Interaction'}
          </button>
          {error && <p className="mt-4 text-red-600 font-medium">{error}</p>}
        </div>

        {prediction && (
          <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">Prediction Results</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-800">Drugs</h3>
                <p className="text-gray-600">{prediction.drug_pair?.[0]} & {prediction.drug_pair?.[1]}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-800">Probability</h3>
                <p className="text-2xl font-bold text-blue-600">{(prediction.interaction_probability * 100).toFixed(2)}%</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-800">Risk Category</h3>
                <p className={`font-semibold ${prediction.risk_category === 'High' ? 'text-red-600' : 'text-green-600'}`}>
                  {prediction.risk_category}
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold text-gray-800 mb-6">Interaction Graph</h2>
          <div className="h-96 border border-gray-200 rounded-lg overflow-hidden">
            <ForceGraph2D graphData={graphData} nodeLabel="id" nodeColor={() => '#3b82f6'} />
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-6">Prediction History</h2>
          <button onClick={handleHistory} className="bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg">
            Load History
          </button>
          {history.length > 0 && (
            <ul className="mt-4 list-disc list-inside">
              {history.map((item, idx) => (
                <li key={idx} className="text-gray-700">
                  {item.drug_pair?.[0]} & {item.drug_pair?.[1]}: {item.probability?.toFixed(2)}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
