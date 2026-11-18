# PathSnap — maps and evaluator UI

## Windows PowerShell setup

Run from the repository root. Use Python 3.12 or later and `uv`.

```powershell
if (!(Test-Path '.venv/Scripts/python.exe')) { uv venv --python (Get-Command python).Source .venv }
uv pip install --python .venv/Scripts/python.exe -r ui/requirements.txt
$env:PYTHONPATH = 'src'
```

The requirements file is UI-specific; leave the model lock and shared package dependencies alone.

## Development map: real training trip

```powershell
.\.venv\Scripts\python.exe -B -m pathsnap.visualization.training_trip --data-root '..\Student folder\Student folder' --trip-id 00000001 --output '..\PathSnap-artifacts\trip-00000001.html'
```

Set `--data-root` to the directory containing `train/` if the dataset is elsewhere. Open the HTML in a browser. It shows blue GPS dots, gray roads near the trip, and the green reference route. The road overlay retains arcs whose endpoints fall in grid cells occupied by GPS/reference points or their neighboring cells; the degree grid is for display only. Opposite-direction road arcs are drawn once in the background layer, while the ordered reference route retains direction and splits at disconnected joins. The raw dots are not joined across gaps. No prediction is included.

The command's small, training-only CLI reader is a temporary adapter for the documented raw files. Use Member 1's reader once its interface is available; the map function already takes GPS longitude/latitude pairs, nodes, directed arcs, and reference arc IDs separately.

## Laptop 1: FastAPI and inference

Member 3 supplies the exact backend startup command, port, artifact paths, and reserved trip setup. The process must listen on the laptop's LAN interface, not just loopback. Keep terminal request logs visible. Run `ipconfig` and use the IPv4 address of the adapter connected to Laptop 2's network. Expose the API only to the intended local network; don't disable the firewall or expose it to the internet.

## Laptop 2: evaluator UI

In the same PowerShell session as setup, use the address and port confirmed by Member 3. The values below are examples, not project defaults:

```powershell
$backendHost = '192.168.1.23'
$backendPort = 8000
$backendUrl = "http://${backendHost}:${backendPort}"
Test-NetConnection -ComputerName $backendHost -Port $backendPort
Invoke-RestMethod -Uri "$backendUrl/health"
.\.venv\Scripts\python.exe -B ui/app.py --backend-url $backendUrl --port 8501
```

On Laptop 2, open `http://127.0.0.1:8501`. The UI binds only to Laptop 2's loopback interface. Its local Python process calls FastAPI on Laptop 1, so browser CORS configuration isn't required for this UI. Restart the UI with a new `--backend-url` if Laptop 1's address changes. If health fails, recheck the address, port, LAN listener, firewall rule, and whether the network isolates clients.

The map uses Folium's default external browser assets and OpenStreetMap tiles. Rehearse on the actual demo network; LAN access to the backend does not guarantee access to these assets. Do not treat the default setup as offline-capable.

## Provisional API adapter — confirm with Member 3

See the [Laptop 1 API handoff](laptop1_api.md) for example responses, route breaks, accuracy, timing units, and error statuses. The adapter is in `ui/app.py` and can be updated once Member 3 confirms the backend contract.
