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

Install the shared project dependencies for the training-trip reader in addition to the UI requirements:

```powershell
uv pip install --python .venv/Scripts/python.exe -r pyproject.toml
.\.venv\Scripts\python.exe -B -m pathsnap.visualization.training_trip --data-root '..\Student folder\Student folder' --trip-id 00000001 --output '..\PathSnap-artifacts\trip-00000001.html'
```

Set `--data-root` to the directory containing `train/` if the dataset is elsewhere. Open the HTML in a browser. It shows blue GPS dots, gray roads near the trip, and the green reference route. The road overlay retains arcs whose endpoints fall in grid cells occupied by GPS/reference points or their neighboring cells; the degree grid is for display only. Opposite-direction road arcs are drawn once in the background layer, while the ordered reference route retains direction and splits at disconnected joins. The raw dots are not joined across gaps. No prediction is included.

The CLI only accepts trips under `train/` and calls the shared `pathsnap.data.load_trip` reader with `include_route=True` for this development map. Inference uses `include_route=False`; the evaluator UI does not read trip files.

## Laptop 1: FastAPI and inference

The [FastAPI server](../README.md#run-the-fastapi-backend-on-laptop-1) is implemented. On a Windows Laptop 1, run these commands from the repository root, replacing the data and trusted fitted model paths with the actual paths on that laptop:

```powershell
uv sync --locked --extra api --extra models
$env:PATHSNAP_DATA_ROOT = 'C:\path\to\student-dataset'
$env:PATHSNAP_MODEL = 'C:\path\to\fitted-model.joblib'
$env:PATHSNAP_TRIP_PARTITION = 'student_test'
uv run --locked --extra api --extra models uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

The data root must contain the `student_test` directory. The process must listen on the LAN interface, not just loopback. Run `ipconfig` and share the IPv4 address of the adapter connected to Laptop 2's network. Expose the API only to the intended local network; don't disable the firewall or expose it to the internet. Keep the terminal logs visible and confirm that the rehearsal log shows both the trip ID and the successful request status; the standard access log doesn't include the request body.

## Laptop 2: evaluator UI

In the same PowerShell session as setup, use the address and port confirmed by Member 3. The values below are examples, not project defaults:

```powershell
$backendHost = '192.168.1.23'
$backendPort = 8000
$backendUrl = "http://${backendHost}:${backendPort}"
Test-NetConnection -ComputerName $backendHost -Port $backendPort
Invoke-RestMethod -Uri "$backendUrl/health"
Invoke-RestMethod -Uri "$backendUrl/trips"
.\.venv\Scripts\python.exe -B ui/app.py --backend-url $backendUrl --port 8501
```

On Laptop 2, open `http://127.0.0.1:8501`. The UI binds only to Laptop 2's loopback interface. Its local Python process calls FastAPI on Laptop 1, so browser CORS configuration isn't required for this UI. Restart the UI with a new `--backend-url` if Laptop 1's address changes. If health fails, recheck the address, port, LAN listener, firewall rule, and whether the network isolates clients.

The map uses Folium's default external browser assets and OpenStreetMap tiles. Rehearse on the actual demo network; LAN access to the backend does not guarantee access to these assets. Do not treat the default setup as offline-capable.

## Implemented API contract

The Laptop 2 adapter in `ui/app.py` matches the FastAPI response models. See the [Laptop 1 API handoff](laptop1_api.md) for example responses, route breaks, accuracy, timing units, and error statuses. The examples are synthetic; verify the actual two-laptop flow with a fitted model and a `student_test` trip before the demo.
