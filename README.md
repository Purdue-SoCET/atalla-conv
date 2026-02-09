## Flow
![diagram](diagram.svg)
- Store IFMap in HWC layout.
- Decompose each `R x S` kernel into `R*S` 1x1 tiles.
- For each tile, stream the HWC word into a weight-stationary systolic array.
- Optionally pack multiple tiles per array to improve utilization.
- Accumulate partial sums into the final OFMap.

## Run
CLI:
```
python im2col.py
```

Streamlit GUI:
```
streamlit run streamlit_app.py
```
