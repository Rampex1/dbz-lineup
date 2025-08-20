import React, { useState, useCallback, useEffect } from 'react';
import { Plus, Trash2, Users, Scale, CheckCircle, AlertCircle, Ship, UserCheck, User, ArrowRight, ArrowLeft, RefreshCw, ToggleRight, Download, FileText } from 'lucide-react';

// Backend API base URL
const API_BASE = 'http://127.0.0.1:5000';

const BoatAssignmentManager = () => {
  const [boats, setBoats] = useState([]);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentBoat, setCurrentBoat] = useState({ size: 20, gender: 'Mixed' });
  const [people, setPeople] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [lineup, setLineup] = useState([]);
  const [isLineupLoading, setIsLineupLoading] = useState(false);
  const [lineupError, setLineupError] = useState(null);

  useEffect(() => {
    loadPeople();
  }, []);

  const loadPeople = async () => {
    try {
      const res = await fetch(`${API_BASE}/people`);
      const data = await res.json();
      setPeople(Array.isArray(data.people) ? data.people : []);
    } catch (e) {
      console.error('Failed to fetch people', e);
      setPeople([]);
    }
  };

  const togglePerson = async (name, currentActiveStatus) => {
    try {
      const res = await fetch(`${API_BASE}/people/${encodeURIComponent(name)}/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to toggle person');
      await loadPeople();
    } catch (e) {
      console.error('Failed to toggle person', e);
    }
  };

  const toggleAllPeople = async () => {
    try {
      const allActive = people.every(p => p.active);
      const res = await fetch(`${API_BASE}/people/toggle-all`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: !allActive })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to toggle all people');
      await loadPeople();
    } catch (e) {
      console.error('Failed to toggle all people', e);
    }
  };

  const scrapeAttendance = async () => {
    try {
      const res = await fetch(`${API_BASE}/people/scrape-attendance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to scrape attendance');
      await loadPeople();
    } catch (e) {
      console.error('Failed to scrape attendance', e);
    }
  };

  const importLineup = async () => {
    try {
      setIsLineupLoading(true);
      setLineupError(null);
      const res = await fetch(`${API_BASE}/people/import-lineup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to import lineup');
      setLineup(data.people || []);
      await loadPeople();
    } catch (e) {
      console.error('Failed to import lineup', e);
      setLineupError(e.message);
    } finally {
      setIsLineupLoading(false);
    }
  };

  const openModal = () => {
    setIsModalOpen(true);
    importLineup();
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setLineup([]);
    setLineupError(null);
  };

  const addBoat = useCallback(() => {
    if (currentBoat.size > 0 && currentBoat.size % 2 === 0) {
      setBoats(prev => [...prev, { ...currentBoat, id: Date.now() }]);
      setCurrentBoat({ size: 20, gender: 'Mixed' });
    }
  }, [currentBoat]);

  const removeBoat = useCallback((id) => {
    setBoats(prev => prev.filter(boat => boat.id !== id));
  }, []);

  const generateAssignments = useCallback(async () => {
    if (boats.length === 0 || people.length === 0) return;

    try {
      setIsLoading(true);
      const payload = { boats: boats.map(b => ({ size: b.size, gender: b.gender })) };
      const res = await fetch(`${API_BASE}/assignments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to create assignments');
      setResults(data.assignments || {});
    } catch (e) {
      console.error(e);
      setResults(null);
    } finally {
      setIsLoading(false);
    }
  }, [boats, people]);

  const getTotalPeople = () => boats.reduce((sum, boat) => sum + boat.size, 0);

  const getGenderIcon = (gender) => {
    switch (gender) {
      case 'Open': return <User className="w-4 h-4" />;
      case 'Women': return <UserCheck className="w-4 h-4" />;
      case 'Mixed': return <Users className="w-4 h-4" />;
      default: return null;
    }
  };

  const validateBoats = () => {
    const totalPeople = getTotalPeople();
    const availablePeople = people.filter(p => p.active).length;
    return totalPeople <= availablePeople;
  };

  const splitBySide = (arr) => ({
    left: arr.filter(p => p.side === 'L'),
    right: arr.filter(p => p.side === 'R'),
    ambi: arr.filter(p => p.side === 'A')
  });

  const getSubstitutionInfo = (boatIdx) => {
    if (!results?.first_half || !results?.second_half) return null;

    const firstHalf = results.first_half[boatIdx];
    const secondHalf = results.second_half[boatIdx];

    if (!firstHalf || !secondHalf) return null;

    const firstHalfPeople = [...firstHalf.left, ...firstHalf.right];
    const secondHalfPeople = [...secondHalf.left, ...secondHalf.right];

    const firstHalfNames = new Set(firstHalfPeople.map(p => p.name));
    const secondHalfNames = new Set(secondHalfPeople.map(p => p.name));

    const switchingOff = firstHalfPeople.filter(p => !secondHalfNames.has(p.name));
    const switchingOn = secondHalfPeople.filter(p => !firstHalfNames.has(p.name));
    const continuing = firstHalfPeople.filter(p => secondHalfNames.has(p.name));

    return { switchingOff, switchingOn, continuing };
  };

  const renderBoatAssignment = (boatIdx, half) => {
    const boat = boats[boatIdx];
    const assignment = results[half][boatIdx];

    if (!assignment || !boat) return null;

    const leftWeight = assignment.left.reduce((sum, p) => sum + p.weight, 0);
    const rightWeight = assignment.right.reduce((sum, p) => sum + p.weight, 0);
    const weightDiff = Math.abs(leftWeight - rightWeight);

    return (
      <div className="grid md:grid-cols-2 gap-6">
        {/* Left Side */}
        <div className="space-y-3">
          <h4 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            Left Side ({assignment.left.length})
          </h4>
          <div className="text-sm text-gray-600 mb-3">
            Total Weight: {leftWeight.toFixed(1)} lbs
          </div>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {assignment.left.map((person, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                <div>
                  <div className="font-medium text-gray-900">{person.name}</div>
                  <div className="text-sm text-gray-600">
                    {person.gender} • {person.side === 'A' ? 'Ambidextrous' : person.side === 'L' ? 'Left-handed' : 'Right-handed'}
                  </div>
                </div>
                <div className="text-sm font-medium text-gray-900">
                  {person.weight} lbs
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Side */}
        <div className="space-y-3">
          <h4 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            Right Side ({assignment.right.length})
          </h4>
          <div className="text-sm text-gray-600 mb-3">
            Total Weight: {rightWeight.toFixed(1)} lbs
          </div>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {assignment.right.map((person, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                <div>
                  <div className="font-medium text-gray-900">{person.name}</div>
                  <div className="text-sm text-gray-600">
                    {person.gender} • {person.side === 'A' ? 'Ambidextrous' : person.side === 'L' ? 'Left-handed' : 'Right-handed'}
                  </div>
                </div>
                <div className="text-sm font-medium text-gray-900">
                  {person.weight} lbs
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderStatistics = (boatIdx, half) => {
    const assignment = results[half][boatIdx];
    if (!assignment) return null;

    const leftWeight = assignment.left.reduce((sum, p) => sum + p.weight, 0);
    const rightWeight = assignment.right.reduce((sum, p) => sum + p.weight, 0);
    const weightDiff = Math.abs(leftWeight - rightWeight);

    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">
            {assignment.left.filter(p => p.gender === 'M').length + assignment.right.filter(p => p.gender === 'M').length}
          </div>
          <div className="text-sm text-gray-600">Males</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-pink-600">
            {assignment.left.filter(p => p.gender === 'F').length + assignment.right.filter(p => p.gender === 'F').length}
          </div>
          <div className="text-sm text-gray-600">Females</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-600">
            {(leftWeight + rightWeight).toFixed(1)}
          </div>
          <div className="text-sm text-gray-600">Total Weight</div>
        </div>
        <div className="text-center">
          <div className={`text-2xl font-bold ${weightDiff < 10 ? 'text-green-600' : weightDiff < 25 ? 'text-yellow-600' : 'text-red-600'}`}>
            {((weightDiff / (leftWeight + rightWeight)) * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Imbalance</div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex justify-center items-center gap-3 mb-4">
            <Ship className="w-10 h-10 text-blue-600" />
            <h1 className="text-4xl font-bold text-gray-900">DBZ Lineup Maker</h1>
          </div>
          <p className="text-lg text-gray-600">Configure boats and optimize weight distribution</p>
        </div>

        {!results ? (
          /* Configuration Phase */
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Boat Builder */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-semibold text-gray-900 mb-6 flex items-center gap-2">
                <Plus className="w-6 h-6 text-blue-600" />
                Add Boat
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Boat Size (must be even)
                  </label>
                  <input
                    type="number"
                    value={currentBoat.size}
                    onChange={(e) => {
                      const raw = e.target.value;
                      if (raw === '') {
                        setCurrentBoat(prev => ({ ...prev, size: '' }));
                        return;
                      }
                      const normalized = raw.replace(/^0+(?=\d)/, '');
                      const parsed = parseInt(normalized, 10);
                      setCurrentBoat(prev => ({ ...prev, size: Number.isNaN(parsed) ? '' : parsed }));
                    }}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    min="2"
                    step="2"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Gender Requirement
                  </label>
                  <select
                    value={currentBoat.gender}
                    onChange={(e) => setCurrentBoat(prev => ({ ...prev, gender: e.target.value }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="Mixed">Mixed (Half Male, Half Female)</option>
                    <option value="Open">Open (Males Only)</option>
                    <option value="Women">Women (Females Only)</option>
                  </select>
                </div>

                <button
                  onClick={addBoat}
                  disabled={currentBoat.size <= 0 || currentBoat.size % 2 !== 0}
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  Add Boat
                </button>
              </div>
            </div>

            {/* Boat List */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-semibold text-gray-900 mb-6 flex items-center gap-2">
                <Ship className="w-6 h-6 text-blue-600" />
                Configured Boats ({boats.length})
              </h2>

              <div className="space-y-3 mb-6 max-h-64 overflow-y-auto">
                {boats.map((boat, index) => (
                  <div key={boat.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center gap-3">
                      {getGenderIcon(boat.gender)}
                      <div>
                        <div className="font-medium text-gray-900">
                          Boat {index + 1}
                        </div>
                        <div className="text-sm text-gray-600">
                          {boat.size} people • {boat.gender}
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => removeBoat(boat.id)}
                      className="text-red-600 hover:text-red-800 transition-colors"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                ))}
                {boats.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    No boats configured yet. Add a boat to get started.
                  </div>
                )}
              </div>

              {boats.length > 0 && (
                <div className="border-t pt-4">
                  <div className="flex justify-between items-center mb-4">
                    <div className="text-sm text-gray-600">
                      Total People Required: <span className="font-semibold">{getTotalPeople()}</span> / {people.filter(p => p.active).length} active
                    </div>
                    {!validateBoats() && (
                      <div className="flex items-center gap-1 text-red-600">
                        <AlertCircle className="w-4 h-4" />
                        <span className="text-sm">Not enough active people</span>
                      </div>
                    )}
                  </div>

                  <button
                    onClick={generateAssignments}
                    disabled={!validateBoats() || isLoading}
                    className="w-full bg-green-600 text-white py-3 px-4 rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
                  >
                    {isLoading ? 'Optimizing Assignments...' : 'Generate Optimal Assignments'}
                  </button>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Results Phase */
          <div className="space-y-6">
            <div className="flex items-center justify-between mb-8">
              <h2 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                <CheckCircle className="w-8 h-8 text-green-600" />
                Assignment Results - Two Halves
              </h2>
              <button
                onClick={() => setResults(null)}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
              >
                Configure New Boats
              </button>
            </div>

            <div className="space-y-8">
              {boats.map((boat, boatIdx) => {
                const substitutionInfo = getSubstitutionInfo(boatIdx);
                const firstHalf = results.first_half?.[boatIdx];
                const secondHalf = results.second_half?.[boatIdx];

                if (!firstHalf || !secondHalf) return null;

                return (
                  <div key={boat.id} className="bg-white rounded-xl shadow-lg overflow-hidden">
                    {/* Boat Header */}
                    <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-6">
                      <div className="flex items-center gap-3">
                        <Ship className="w-8 h-8" />
                        <div>
                          <h3 className="text-2xl font-bold">Boat {boatIdx + 1}</h3>
                          <p className="text-blue-100">{boat.gender} • {boat.size} people</p>
                        </div>
                      </div>
                    </div>

                    <div className="p-6 space-y-8">
                      {/* First Half */}
                      <div>
                        <div className="flex items-center gap-2 mb-4">
                          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                          <h4 className="text-xl font-bold text-gray-900">First Half</h4>
                          <div className="flex items-center gap-2 text-sm text-gray-600 ml-4">
                            <Scale className="w-4 h-4" />
                            <span>Weight Diff: {Math.abs(
                              firstHalf.left.reduce((sum, p) => sum + p.weight, 0) -
                              firstHalf.right.reduce((sum, p) => sum + p.weight, 0)
                            ).toFixed(1)} lbs</span>
                          </div>
                        </div>

                        {renderBoatAssignment(boatIdx, 'first_half')}

                        <div className="mt-6 pt-6 border-t border-gray-200">
                          {renderStatistics(boatIdx, 'first_half')}
                        </div>
                      </div>

                      {/* Substitution Section */}
                      {substitutionInfo && (substitutionInfo.switchingOff.length > 0 || substitutionInfo.switchingOn.length > 0) && (
                        <div className="bg-gray-50 rounded-lg p-6">
                          <div className="flex items-center gap-2 mb-4">
                            <RefreshCw className="w-5 h-5 text-orange-600" />
                            <h4 className="text-lg font-bold text-gray-900">Substitutions</h4>
                          </div>

                          <div className="grid md:grid-cols-2 gap-6">
                            {/* Switching Off */}
                            <div>
                              <div className="flex items-center gap-2 mb-3">
                                <ArrowLeft className="w-4 h-4 text-red-600" />
                                <h5 className="font-semibold text-gray-900">Switching Off ({substitutionInfo.switchingOff.length})</h5>
                              </div>
                              <div className="space-y-2">
                                {substitutionInfo.switchingOff.map((person, idx) => (
                                  <div key={idx} className="flex items-center justify-between p-3 bg-red-50 rounded-lg border border-red-100">
                                    <div>
                                      <div className="font-medium text-gray-900">{person.name}</div>
                                      <div className="text-sm text-gray-600">
                                        {person.gender} • {person.side === 'A' ? 'Ambidextrous' : person.side === 'L' ? 'Left-handed' : 'Right-handed'}
                                      </div>
                                    </div>
                                    <div className="text-sm font-medium text-gray-900">
                                      {person.weight} lbs
                                    </div>
                                  </div>
                                ))}
                                {substitutionInfo.switchingOff.length === 0 && (
                                  <div className="text-gray-500 text-sm italic">No substitutions out</div>
                                )}
                              </div>
                            </div>

                            {/* Switching On */}
                            <div>
                              <div className="flex items-center gap-2 mb-3">
                                <ArrowRight className="w-4 h-4 text-green-600" />
                                <h5 className="font-semibold text-gray-900">Switching On ({substitutionInfo.switchingOn.length})</h5>
                              </div>
                              <div className="space-y-2">
                                {substitutionInfo.switchingOn.map((person, idx) => (
                                  <div key={idx} className="flex items-center justify-between p-3 bg-green-50 rounded-lg border border-green-100">
                                    <div>
                                      <div className="font-medium text-gray-900">{person.name}</div>
                                      <div className="text-sm text-gray-600">
                                        {person.gender} • {person.side === 'A' ? 'Ambidextrous' : person.side === 'L' ? 'Left-handed' : 'Right-handed'}
                                      </div>
                                    </div>
                                    <div className="text-sm font-medium text-gray-900">
                                      {person.weight} lbs
                                    </div>
                                  </div>
                                ))}
                                {substitutionInfo.switchingOn.length === 0 && (
                                  <div className="text-gray-500 text-sm italic">No substitutions in</div>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Second Half */}
                      <div>
                        <div className="flex items-center gap-2 mb-4">
                          <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                          <h4 className="text-xl font-bold text-gray-900">Second Half</h4>
                          <div className="flex items-center gap-2 text-sm text-gray-600 ml-4">
                            <Scale className="w-4 h-4" />
                            <span>Weight Diff: {Math.abs(
                              secondHalf.left.reduce((sum, p) => sum + p.weight, 0) -
                              secondHalf.right.reduce((sum, p) => sum + p.weight, 0)
                            ).toFixed(1)} lbs</span>
                          </div>
                        </div>

                        {renderBoatAssignment(boatIdx, 'second_half')}

                        <div className="mt-6 pt-6 border-t border-gray-200">
                          {renderStatistics(boatIdx, 'second_half')}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Full Roster */}
        <div className="mt-16 border-t border-gray-200 pt-16">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Team Roster</h2>
            <p className="text-lg text-gray-600">Complete list of available team members</p>
            <div className="flex justify-center gap-4 mt-4">
              <button
                onClick={toggleAllPeople}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
              >
                <ToggleRight className="w-5 h-5" />
                {people.every(p => p.active) ? 'Deactivate All' : 'Activate All'}
              </button>
              <button
                onClick={scrapeAttendance}
                className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
              >
                <Download className="w-5 h-5" />
                Scrape Google Sheets
              </button>
              <button
                onClick={openModal}
                className="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 transition-colors flex items-center gap-2"
              >
                <FileText className="w-5 h-5" />
                Import Lineup
              </button>
            </div>
          </div>

          {/* Modal */}
          {isModalOpen && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-white rounded-xl shadow-2xl p-6 max-w-4xl w-full max-h-[80vh] overflow-y-auto">
                <div className="flex items-center gap-3 mb-6">
                  <FileText className="w-8 h-8 text-indigo-600" />
                  <h2 className="text-2xl font-bold text-gray-900">Imported Lineup</h2>
                </div>

                {isLineupLoading ? (
                  <div className="text-center py-12">
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-600 mx-auto"></div>
                    <p className="mt-4 text-gray-600">Loading lineup...</p>
                  </div>
                ) : lineupError ? (
                  <div className="text-center py-12 text-red-600">
                    <AlertCircle className="w-12 h-12 mx-auto mb-4" />
                    <p>{lineupError}</p>
                  </div>
                ) : lineup.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="bg-indigo-50">
                          <th className="px-4 py-3 text-sm font-semibold text-gray-900 border-b">Name</th>
                          <th className="px-4 py-3 text-sm font-semibold text-gray-900 border-b">Gender</th>
                          <th className="px-4 py-3 text-sm font-semibold text-gray-900 border-b">Side</th>
                          <th className="px-4 py-3 text-sm font-semibold text-gray-900 border-b">Weight (lbs)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {lineup.map((person, idx) => (
                          <tr key={idx} className="border-b hover:bg-gray-50">
                            <td className="px-4 py-3 text-gray-900">{person.name}</td>
                            <td className="px-4 py-3 text-gray-600">{person.gender === 'M' ? 'Male' : 'Female'}</td>
                            <td className="px-4 py-3 text-gray-600">
                              {person.side === 'A' ? 'Ambidextrous' : person.side === 'L' ? 'Left-handed' : 'Right-handed'}
                            </td>
                            <td className="px-4 py-3 text-gray-600">{person.weight}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="text-center py-12 text-gray-600">
                    <p>No lineup data available</p>
                  </div>
                )}

                <div className="mt-6 flex justify-end">
                  <button
                    onClick={closeModal}
                    className="bg-gray-600 text-white px-6 py-2 rounded-lg hover:bg-gray-700 transition-colors"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          )}

          {people.length === 0 ? (
            <div className="bg-white rounded-xl shadow-lg p-12 text-center">
              <Users className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <div className="text-xl text-gray-600">No team members loaded</div>
            </div>
          ) : (
            <div className="grid lg:grid-cols-2 gap-12">
              {/* Men's Team */}
              {(() => {
                const guys = people.filter(p => p.gender === 'M');
                const groups = splitBySide(guys);
                return (
                  <div className="bg-white rounded-xl shadow-lg overflow-hidden">
                    <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-8">
                      <div className="flex items-center gap-3">
                        <User className="w-8 h-8" />
                        <div>
                          <h3 className="text-2xl font-bold">Men's Team</h3>
                          <p className="text-blue-100">{guys.length} athletes ({guys.filter(p => p.active).length} active)</p>
                        </div>
                      </div>
                    </div>

                    <div className="p-8">
                      <div className="grid gap-8">
                        {/* Left-handed */}
                        <div>
                          <div className="flex items-center gap-2 mb-6">
                            <div className="w-3 h-3 bg-indigo-500 rounded-full"></div>
                            <h4 className="text-lg font-semibold text-gray-900">Left-handed ({groups.left.length})</h4>
                          </div>
                          {groups.left.length > 0 ? (
                            <div className="grid sm:grid-cols-2 gap-4">
                              {groups.left.map((p, idx) => (
                                <div key={`g-l-${idx}`} className="bg-indigo-50 rounded-lg p-4 border border-indigo-100 hover:shadow-md transition-shadow flex justify-between items-center">
                                  <div>
                                    <div className="font-semibold text-gray-900 text-base mb-1">{p.name}</div>
                                    <div className="text-sm text-gray-600 font-medium">{p.weight} lbs</div>
                                  </div>
                                  <label className="relative inline-flex items-center cursor-pointer">
                                    <input
                                      type="checkbox"
                                      checked={p.active}
                                      onChange={() => togglePerson(p.name, p.active)}
                                      className="sr-only peer"
                                    />
                                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer peer-checked:bg-blue-600"></div>
                                    <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform peer-checked:translate-x-5"></div>
                                  </label>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="text-gray-500 bg-gray-50 rounded-lg p-6 text-center italic">No left-handed athletes</div>
                          )}
                        </div>

                        {/* Right-handed */}
                        <div>
                          <div className="flex items-center gap-2 mb-6">
                            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                            <h4 className="text-lg font-semibold text-gray-900">Right-handed ({groups.right.length})</h4>
                          </div>
                          {groups.right.length > 0 ? (
                            <div className="grid sm:grid-cols-2 gap-4">
                              {groups.right.map((p, idx) => (
                                <div key={`g-r-${idx}`} className="bg-blue-50 rounded-lg p-4 border border-blue-100 hover:shadow-md transition-shadow flex justify-between items-center">
                                  <div>
                                    <div className="font-semibold text-gray-900 text-base mb-1">{p.name}</div>
                                    <div className="text-sm text-gray-600 font-medium">{p.weight} lbs</div>
                                  </div>
                                  <label className="relative inline-flex items-center cursor-pointer">
                                    <input
                                      type="checkbox"
                                      checked={p.active}
                                      onChange={() => togglePerson(p.name, p.active)}
                                      className="sr-only peer"
                                    />
                                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer peer-checked:bg-blue-600"></div>
                                    <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform peer-checked:translate-x-5"></div>
                                  </label>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="text-gray-500 bg-gray-50 rounded-lg p-6 text-center italic">No right-handed athletes</div>
                          )}
                        </div>

                        {/* Ambidextrous */}
                        <div>
                          <div className="flex items-center gap-2 mb-6">
                            <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                            <h4 className="text-lg font-semibold text-gray-900">Ambidextrous ({groups.ambi.length})</h4>
                          </div>
                          {groups.ambi.length > 0 ? (
                            <div className="grid sm:grid-cols-2 gap-4">
                              {groups.ambi.map((p, idx) => (
                                <div key={`g-a-${idx}`} className="bg-purple-50 rounded-lg p-4 border border-purple-100 hover:shadow-md transition-shadow flex justify-between items-center">
                                  <div>
                                    <div className="font-semibold text-gray-900 text-base mb-1">{p.name}</div>
                                    <div className="text-sm text-gray-600 font-medium">{p.weight} lbs</div>
                                  </div>
                                  <label className="relative inline-flex items-center cursor-pointer">
                                    <input
                                      type="checkbox"
                                      checked={p.active}
                                      onChange={() => togglePerson(p.name, p.active)}
                                      className="sr-only peer"
                                    />
                                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer peer-checked:bg-blue-600"></div>
                                    <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform peer-checked:translate-x-5"></div>
                                  </label>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="text-gray-500 bg-gray-50 rounded-lg p-6 text-center italic">No ambidextrous athletes</div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })()}

              {/* Women's Team */}
              {(() => {
                const girls = people.filter(p => p.gender === 'F');
                const groups = splitBySide(girls);
                return (
                  <div className="bg-white rounded-xl shadow-lg overflow-hidden">
                    <div className="bg-gradient-to-r from-pink-600 to-rose-600 text-white p-8">
                      <div className="flex items-center gap-3">
                        <UserCheck className="w-8 h-8" />
                        <div>
                          <h3 className="text-2xl font-bold">Women's Team</h3>
                          <p className="text-pink-100">{girls.length} athletes ({girls.filter(p => p.active).length} active)</p>
                        </div>
                      </div>
                    </div>

                    <div className="p-8">
                      <div className="grid gap-8">
                        {/* Left-handed */}
                        <div>
                          <div className="flex items-center gap-2 mb-6">
                            <div className="w-3 h-3 bg-rose-500 rounded-full"></div>
                            <h4 className="text-lg font-semibold text-gray-900">Left-handed ({groups.left.length})</h4>
                          </div>
                          {groups.left.length > 0 ? (
                            <div className="grid sm:grid-cols-2 gap-4">
                              {groups.left.map((p, idx) => (
                                <div key={`f-l-${idx}`} className="bg-rose-50 rounded-lg p-4 border border-rose-100 hover:shadow-md transition-shadow flex justify-between items-center">
                                  <div>
                                    <div className="font-semibold text-gray-900 text-base mb-1">{p.name}</div>
                                    <div className="text-sm text-gray-600 font-medium">{p.weight} lbs</div>
                                  </div>
                                  <label className="relative inline-flex items-center cursor-pointer">
                                    <input
                                      type="checkbox"
                                      checked={p.active}
                                      onChange={() => togglePerson(p.name, p.active)}
                                      className="sr-only peer"
                                    />
                                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer peer-checked:bg-blue-600"></div>
                                    <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform peer-checked:translate-x-5"></div>
                                  </label>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="text-gray-500 bg-gray-50 rounded-lg p-6 text-center italic">No left-handed athletes</div>
                          )}
                        </div>

                        {/* Right-handed */}
                        <div>
                          <div className="flex items-center gap-2 mb-6">
                            <div className="w-3 h-3 bg-pink-500 rounded-full"></div>
                            <h4 className="text-lg font-semibold text-gray-900">Right-handed ({groups.right.length})</h4>
                          </div>
                          {groups.right.length > 0 ? (
                            <div className="grid sm:grid-cols-2 gap-4">
                              {groups.right.map((p, idx) => (
                                <div key={`f-r-${idx}`} className="bg-pink-50 rounded-lg p-4 border border-pink-100 hover:shadow-md transition-shadow flex justify-between items-center">
                                  <div>
                                    <div className="font-semibold text-gray-900 text-base mb-1">{p.name}</div>
                                    <div className="text-sm text-gray-600 font-medium">{p.weight} lbs</div>
                                  </div>
                                  <label className="relative inline-flex items-center cursor-pointer">
                                    <input
                                      type="checkbox"
                                      checked={p.active}
                                      onChange={() => togglePerson(p.name, p.active)}
                                      className="sr-only peer"
                                    />
                                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer peer-checked:bg-blue-600"></div>
                                    <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform peer-checked:translate-x-5"></div>
                                  </label>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="text-gray-500 bg-gray-50 rounded-lg p-6 text-center italic">No right-handed athletes</div>
                          )}
                        </div>

                        {/* Ambidextrous */}
                        <div>
                          <div className="flex items-center gap-2 mb-6">
                            <div className="w-3 h-3 bg-violet-500 rounded-full"></div>
                            <h4 className="text-lg font-semibold text-gray-900">Ambidextrous ({groups.ambi.length})</h4>
                          </div>
                          {groups.ambi.length > 0 ? (
                            <div className="grid sm:grid-cols-2 gap-4">
                              {groups.ambi.map((p, idx) => (
                                <div key={`f-a-${idx}`} className="bg-violet-50 rounded-lg p-4 border border-violet-100 hover:shadow-md transition-shadow flex justify-between items-center">
                                  <div>
                                    <div className="font-semibold text-gray-900 text-base mb-1">{p.name}</div>
                                    <div className="text-sm text-gray-600 font-medium">{p.weight} lbs</div>
                                  </div>
                                  <label className="relative inline-flex items-center cursor-pointer">
                                    <input
                                      type="checkbox"
                                      checked={p.active}
                                      onChange={() => togglePerson(p.name, p.active)}
                                      className="sr-only peer"
                                    />
                                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full peer peer-checked:bg-blue-600"></div>
                                    <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform peer-checked:translate-x-5"></div>
                                  </label>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="text-gray-500 bg-gray-50 rounded-lg p-6 text-center italic">No ambidextrous athletes</div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BoatAssignmentManager;