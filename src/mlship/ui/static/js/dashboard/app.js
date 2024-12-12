function App() {
    const [serverStatus, setServerStatus] = React.useState(null);
    const [modelPath, setModelPath] = React.useState('');
    const [loading, setLoading] = React.useState(false);
    const [error, setError] = React.useState(null);

    // Fetch server status periodically
    React.useEffect(() => {
        const fetchStatus = async () => {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                setServerStatus(data);
                setError(null);
            } catch (err) {
                setServerStatus(null);
                setError('Server is not running');
            }
        };

        fetchStatus();
        const interval = setInterval(fetchStatus, 5000);
        return () => clearInterval(interval);
    }, []);

    // Handle model deployment
    const handleDeploy = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/deploy', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_path: modelPath }),
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to deploy model');
            }

            const data = await response.json();
            setModelPath('');
            setError(null);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container mx-auto px-4 py-8">
            <header className="mb-8">
                <h1 className="text-3xl font-bold text-gray-900">MLship Dashboard</h1>
                <p className="text-gray-600">Deploy and monitor your ML models</p>
            </header>

            {/* Server Status */}
            <div className="mb-8 p-6 bg-white rounded-lg shadow">
                <h2 className="text-xl font-semibold mb-4">Server Status</h2>
                {error ? (
                    <div className="text-red-600">{error}</div>
                ) : serverStatus ? (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                            <div className="text-sm text-gray-600">Status</div>
                            <div className="text-lg font-semibold text-green-600">Running</div>
                            <div className="text-sm text-gray-500">PID: {serverStatus.pid}</div>
                        </div>
                        <div>
                            <div className="text-sm text-gray-600">Memory Usage</div>
                            <div className="text-lg font-semibold">{serverStatus.memory_mb.toFixed(1)} MB</div>
                        </div>
                        <div>
                            <div className="text-sm text-gray-600">CPU Usage</div>
                            <div className="text-lg font-semibold">{serverStatus.cpu_percent.toFixed(1)}%</div>
                        </div>
                        {serverStatus.metrics && (
                            <>
                                <div>
                                    <div className="text-sm text-gray-600">Total Requests</div>
                                    <div className="text-lg font-semibold">{serverStatus.metrics.requests}</div>
                                </div>
                                <div>
                                    <div className="text-sm text-gray-600">Average Latency</div>
                                    <div className="text-lg font-semibold">{serverStatus.metrics.avg_latency.toFixed(2)} ms</div>
                                </div>
                            </>
                        )}
                    </div>
                ) : (
                    <div className="text-gray-600">Loading status...</div>
                )}
            </div>

            {/* Deploy Form */}
            <div className="p-6 bg-white rounded-lg shadow">
                <h2 className="text-xl font-semibold mb-4">Deploy Model</h2>
                <form onSubmit={handleDeploy}>
                    <div className="mb-4">
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Model Path
                        </label>
                        <input
                            type="text"
                            value={modelPath}
                            onChange={(e) => setModelPath(e.target.value)}
                            placeholder="/path/to/your/model.joblib"
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                            required
                        />
                    </div>
                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
                    >
                        {loading ? 'Deploying...' : 'Deploy Model'}
                    </button>
                </form>
            </div>
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById('root')); 