import React, { useState, useEffect } from 'react';
import axios from 'axios';
import AsyncSelect from 'react-select/async';

function DrugCard({ drug }) {
    if (!drug) return null;
    return (
      <div className="mt-2 p-3 bg-slate-50 border border-slate-100 rounded-lg">
        <h3 className="text-sm font-semibold text-slate-800">{drug.label}</h3>
        <p className="text-xs text-slate-500">ID: {drug.value}</p>
      </div>
    );
  }

function App() {
  const [drugA, setDrugA] = useState(null);
  const [drugB, setDrugB] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
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
      return response.data.results.map(drug => ({ value: drug.id || drug.name, label: drug.name || drug.id }));
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
    <div className="min-h-screen relative overflow-hidden bg-gradient-to-br from-[#f6f4ef] via-white to-[#e8f3ef]">
      <div className="absolute -top-24 -right-20 h-80 w-80 rounded-full bg-emerald-200/40 blur-3xl" />
      <div className="absolute -bottom-32 -left-24 h-96 w-96 rounded-full bg-amber-200/30 blur-3xl" />
      <div className="container mx-auto px-4 py-10 max-w-6xl relative">
        <header className="mb-10">
          <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-emerald-700 font-semibold">Clinical DDI Console</p>
              <h1 className="text-4xl md:text-5xl font-semibold text-slate-900 mt-2 font-display">Drug–Drug Interaction Intelligence</h1>
              <p className="text-slate-600 mt-3 max-w-2xl">
                Evidence‑grounded interaction assessment with enforced clinical risk rules and auditability.
              </p>
            </div>
            <div className="text-xs text-slate-500">
              Session ID: <span className="font-mono">{userId}</span>
            </div>
          </div>
        </header>

        <div className="bg-white/90 backdrop-blur rounded-2xl shadow-xl p-6 mb-8 border border-slate-100">
          <h2 className="text-xl font-semibold text-slate-900 mb-6 font-display">Select Drugs</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Drug A</label>
              <AsyncSelect loadOptions={loadOptions} onChange={setDrugA} placeholder="Search by name or DrugBank ID" />
              <DrugCard drug={drugA} />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Drug B</label>
              <AsyncSelect loadOptions={loadOptions} onChange={setDrugB} placeholder="Search by name or DrugBank ID" />
              <DrugCard drug={drugB} />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">User ID</label>
              <input
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-200"
                placeholder="user_..."
              />
              <p className="text-xs text-slate-500 mt-2">Used for audit trail and history retrieval.</p>
            </div>
          </div>
          <button onClick={handlePredict} disabled={loading}
            className="w-full bg-slate-900 text-emerald-50 font-semibold py-3 px-6 rounded-lg hover:bg-slate-800 disabled:opacity-50">
            {loading ? 'Analyzing...' : 'Predict Interaction'}
          </button>
          {error && <p className="mt-4 text-red-600 font-medium">{error}</p>}
        </div>

        {prediction && (
          <div className="bg-white/95 backdrop-blur rounded-2xl shadow-xl p-6 mb-8 border border-slate-100">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
              <h2 className="text-2xl font-semibold text-slate-900 font-display">Clinical Interaction Summary</h2>
              <div className="text-xs text-slate-500">
                Audit Hash: <span className="font-mono">{prediction.audit_tx_hash || 'pending'}</span>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                <h3 className="font-semibold text-slate-800">Drugs</h3>
                <p className="text-slate-600">{prediction.drug_pair?.[0]} & {prediction.drug_pair?.[1]}</p>
                <p className="text-xs text-slate-500 mt-2">Model Used: {prediction.model_used ? 'Yes' : 'No'}</p>
              </div>
              <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                <h3 className="font-semibold text-slate-800">Interaction Probability</h3>
                <p className="text-3xl font-semibold text-emerald-700">
                  {(prediction.interaction_probability * 100).toFixed(1)}%
                </p>
                <p className="text-xs text-slate-500">Risk: {prediction.risk_level} ({prediction.risk_category})</p>
              </div>
              <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                <h3 className="font-semibold text-slate-800">Recommendation Score</h3>
                <p className="text-3xl font-semibold text-slate-900">{prediction.recommendation_score?.score}</p>
                <p className="text-sm text-slate-600">{prediction.recommendation_score?.category}</p>
              </div>
            </div>

            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-emerald-50 border border-emerald-100 rounded-lg p-4">
                <h3 className="font-semibold text-emerald-900">Clinical Recommendation</h3>
                <p className="text-emerald-800 mt-2">{prediction.recommendation}</p>
                <p className="text-sm text-emerald-800 mt-3">Monitoring: {prediction.monitoring}</p>
                <p className="text-sm text-emerald-800">Dose Adjustment: {prediction.dose_adjustment}</p>
                <p className="text-sm text-emerald-800">Patient Counseling: {prediction.counseling}</p>
              </div>
              <div className="bg-slate-50 border border-slate-100 rounded-lg p-4">
                <h3 className="font-semibold text-slate-900">Mechanism Summary</h3>
                <p className="text-slate-700 mt-2 whitespace-pre-line">
                  {prediction.explanation?.llm_summary || 'No explanation available.'}
                </p>
                <p className="text-xs text-slate-500 mt-3">
                  Flags: {prediction.clinical_flags?.length ? prediction.clinical_flags.join(', ') : 'None'}
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="bg-white/95 backdrop-blur rounded-2xl shadow-xl p-6">
          <h2 className="text-2xl font-semibold text-slate-900 mb-6 font-display">Prediction History</h2>
          <button onClick={handleHistory} className="bg-slate-900 text-emerald-50 font-semibold py-2 px-4 rounded-lg">
            Load History
          </button>
          {history.length > 0 && (
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-500 border-b">
                    <th className="py-2">Drugs</th>
                    <th className="py-2">Probability</th>
                    <th className="py-2">Risk</th>
                    <th className="py-2">Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((item, idx) => (
                    <tr key={idx} className="border-b last:border-0">
                      <td className="py-2 text-slate-700">
                        {(item.drug_names && item.drug_names.join(' & ')) || (item.drug_a_id + ' & ' + item.drug_b_id)}
                      </td>
                      <td className="py-2 text-slate-700">
                        {((item.interaction_probability ?? item.probability) * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 text-slate-700">{item.risk_category || item.risk_level}</td>
                      <td className="py-2 text-slate-500">{item.timestamp || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
