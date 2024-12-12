const CLOUD_PROVIDERS = {
    aws: {
        name: 'Amazon Web Services',
        logo: 'https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg',
        instances: [
            { type: 'g4dn.xlarge', name: 'NVIDIA T4 GPU' },
            { type: 'p3.2xlarge', name: 'NVIDIA V100 GPU' },
            { type: 'p4d.24xlarge', name: 'NVIDIA A100 GPU' }
        ],
        regions: [
            { id: 'us-east-1', name: 'US East (N. Virginia)' },
            { id: 'us-west-2', name: 'US West (Oregon)' },
            { id: 'eu-west-1', name: 'Europe (Ireland)' }
        ]
    },
    gcp: {
        name: 'Google Cloud Platform',
        logo: 'https://upload.wikimedia.org/wikipedia/commons/0/01/Google-cloud-platform.svg',
        instances: [
            { type: 'n1-standard-4-nvidia-tesla-t4', name: 'NVIDIA T4 GPU' },
            { type: 'n1-standard-8-nvidia-tesla-v100', name: 'NVIDIA V100 GPU' },
            { type: 'a2-highgpu-1g', name: 'NVIDIA A100 GPU' }
        ],
        regions: [
            { id: 'us-central1', name: 'US Central (Iowa)' },
            { id: 'us-east4', name: 'US East (N. Virginia)' },
            { id: 'europe-west4', name: 'Europe (Netherlands)' }
        ]
    },
    azure: {
        name: 'Microsoft Azure',
        logo: 'https://upload.wikimedia.org/wikipedia/commons/f/fa/Microsoft_Azure.svg',
        instances: [
            { type: 'Standard_NC4as_T4_v3', name: 'NVIDIA T4 GPU' },
            { type: 'Standard_NC6s_v3', name: 'NVIDIA V100 GPU' },
            { type: 'Standard_ND96asr_v4', name: 'NVIDIA A100 GPU' }
        ],
        regions: [
            { id: 'eastus', name: 'East US' },
            { id: 'westus2', name: 'West US 2' },
            { id: 'westeurope', name: 'West Europe' }
        ]
    }
};

function App() {
    const [selectedProvider, setSelectedProvider] = React.useState(null);
    const [selectedInstance, setSelectedInstance] = React.useState('');
    const [selectedRegion, setSelectedRegion] = React.useState('');
    const [deploying, setDeploying] = React.useState(false);
    const [error, setError] = React.useState(null);
    const [deploymentStatus, setDeploymentStatus] = React.useState(null);

    const handleDeploy = async (e) => {
        e.preventDefault();
        setDeploying(true);
        setError(null);
        setDeploymentStatus(null);

        try {
            const response = await fetch('/api/cloud/deploy', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    provider: selectedProvider,
                    instance_type: selectedInstance,
                    region: selectedRegion,
                }),
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to deploy model');
            }

            const data = await response.json();
            setDeploymentStatus(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setDeploying(false);
        }
    };

    return (
        <div className="container mx-auto px-4 py-8">
            <header className="mb-8">
                <h1 className="text-3xl font-bold text-gray-900">MLship Cloud Deployment</h1>
                <p className="text-gray-600">Deploy your model to cloud GPU instances</p>
            </header>

            {/* Model Info */}
            <div className="mb-8 p-6 bg-white rounded-lg shadow">
                <h2 className="text-xl font-semibold mb-4">Model Information</h2>
                <div className="text-gray-600">
                    <p>Model Path: {window.MODEL_PATH || 'No model specified'}</p>
                </div>
            </div>

            {/* Cloud Provider Selection */}
            <div className="mb-8 p-6 bg-white rounded-lg shadow">
                <h2 className="text-xl font-semibold mb-4">Select Cloud Provider</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {Object.entries(CLOUD_PROVIDERS).map(([id, provider]) => (
                        <button
                            key={id}
                            onClick={() => {
                                setSelectedProvider(id);
                                setSelectedInstance('');
                                setSelectedRegion('');
                            }}
                            className={`p-4 border rounded-lg text-center transition-colors ${
                                selectedProvider === id
                                    ? 'border-indigo-500 bg-indigo-50'
                                    : 'border-gray-200 hover:border-indigo-500'
                            }`}
                        >
                            <img
                                src={provider.logo}
                                alt={provider.name}
                                className="h-12 mx-auto mb-2"
                            />
                            <div className="font-medium">{provider.name}</div>
                        </button>
                    ))}
                </div>
            </div>

            {selectedProvider && (
                <form onSubmit={handleDeploy} className="space-y-8">
                    {/* Instance Selection */}
                    <div className="p-6 bg-white rounded-lg shadow">
                        <h2 className="text-xl font-semibold mb-4">Select Instance Type</h2>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            {CLOUD_PROVIDERS[selectedProvider].instances.map(instance => (
                                <label
                                    key={instance.type}
                                    className={`p-4 border rounded-lg cursor-pointer ${
                                        selectedInstance === instance.type
                                            ? 'border-indigo-500 bg-indigo-50'
                                            : 'border-gray-200'
                                    }`}
                                >
                                    <input
                                        type="radio"
                                        name="instance"
                                        value={instance.type}
                                        checked={selectedInstance === instance.type}
                                        onChange={e => setSelectedInstance(e.target.value)}
                                        className="sr-only"
                                    />
                                    <div className="font-medium">{instance.name}</div>
                                    <div className="text-sm text-gray-500">{instance.type}</div>
                                </label>
                            ))}
                        </div>
                    </div>

                    {/* Region Selection */}
                    <div className="p-6 bg-white rounded-lg shadow">
                        <h2 className="text-xl font-semibold mb-4">Select Region</h2>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            {CLOUD_PROVIDERS[selectedProvider].regions.map(region => (
                                <label
                                    key={region.id}
                                    className={`p-4 border rounded-lg cursor-pointer ${
                                        selectedRegion === region.id
                                            ? 'border-indigo-500 bg-indigo-50'
                                            : 'border-gray-200'
                                    }`}
                                >
                                    <input
                                        type="radio"
                                        name="region"
                                        value={region.id}
                                        checked={selectedRegion === region.id}
                                        onChange={e => setSelectedRegion(e.target.value)}
                                        className="sr-only"
                                    />
                                    <div className="font-medium">{region.name}</div>
                                    <div className="text-sm text-gray-500">{region.id}</div>
                                </label>
                            ))}
                        </div>
                    </div>

                    {/* Deploy Button */}
                    <div className="flex justify-end">
                        <button
                            type="submit"
                            disabled={!selectedInstance || !selectedRegion || deploying}
                            className="px-6 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
                        >
                            {deploying ? 'Deploying...' : 'Deploy to Cloud'}
                        </button>
                    </div>
                </form>
            )}

            {/* Error Message */}
            {error && (
                <div className="mt-8 p-4 bg-red-50 border border-red-200 rounded-lg text-red-600">
                    {error}
                </div>
            )}

            {/* Deployment Status */}
            {deploymentStatus && (
                <div className="mt-8 p-6 bg-green-50 border border-green-200 rounded-lg">
                    <h3 className="text-lg font-semibold text-green-800 mb-2">
                        {deploymentStatus.message}
                    </h3>
                    <div className="text-green-700">
                        <p>Provider: {deploymentStatus.details.provider}</p>
                        <p>Instance: {deploymentStatus.details.instance_type}</p>
                        <p>Region: {deploymentStatus.details.region}</p>
                    </div>
                </div>
            )}
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById('root')); 