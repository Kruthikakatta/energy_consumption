import axios from 'axios';

const API_BASE_URL = 'https://energy-consumption-dpsl.onrender.com';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const predictSimple = async (data) => {
  try {
    const response = await api.post('/predict/simple', data);
    return response.data;
  } catch (error) {
    throw error.response?.data || error.message;
  }
};

export const predictDaily = async (data) => {
  try {
    const response = await api.post('/predict/daily', data);
    return response.data;
  } catch (error) {
    throw error.response?.data || error.message;
  }
};

export const compareScenarios = async (scenario1, scenario2) => {
  try {
    const response = await api.post('/compare', null, {
      params: { scenario_1: scenario1, scenario_2: scenario2 }
    });
    return response.data;
  } catch (error) {
    throw error.response?.data || error.message;
  }
};

export const getModelInfo = async () => {
  try {
    const response = await api.get('/model/info');
    return response.data;
  } catch (error) {
    throw error.response?.data || error.message;
  }
};

export const healthCheck = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw error.response?.data || error.message;
  }
};