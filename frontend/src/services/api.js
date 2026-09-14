const request = async (path, options = {}) => {
  const response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`)
  return body
}

export const getHealth = () => request('/api/health')
export const getModel = () => request('/api/model')
export const getGates = () => request('/api/gates')
export const getExperiments = () => request('/api/experiments')
export const predict = (payload) => request('/api/predict', { method: 'POST', body: JSON.stringify(payload) })
