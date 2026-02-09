import numpy as np
import pandas as pd
import streamlit as st

from im2col import SimConfig, ImplicitIm2colSystolicSim, direct_conv_hwc


def _array_to_df(arr):
    return pd.DataFrame(arr)


def _list_to_df(rows, columns=None):
    return pd.DataFrame(rows, columns=columns)


st.set_page_config(page_title="Implicit Im2col Systolic Sim", layout="wide")
#st.title("Implicit Im2col (Channel-First) Systolic Simulation")

with st.sidebar:
    st.header("Config")
    N = st.number_input("N (batch)", min_value=1, value=1, step=1)
    H = st.number_input("H", min_value=1, value=4, step=1)
    W = st.number_input("W", min_value=1, value=4, step=1)
    C = st.number_input("C (Cin)", min_value=1, value=3, step=1)
    K = st.number_input("K (Cout)", min_value=1, value=2, step=1)
    R = st.number_input("R (kernel height)", min_value=1, value=2, step=1)
    S = st.number_input("S (kernel width)", min_value=1, value=2, step=1)
    array_size = st.number_input("Systolic array size", min_value=1, value=4, step=1)
    stride = st.number_input("Stride", min_value=1, value=1, step=1)
    pad = st.number_input("Pad", min_value=0, value=0, step=1)
    seed = st.number_input("Seed", min_value=0, value=0, step=1)
    use_sequential_init = st.checkbox("Sequential init (deterministic)", value=True)
    max_logs = st.number_input("Max iterations to log", min_value=0, value=32, step=1)

    run = st.button("Run simulation", type="primary")


if run:
    cfg = SimConfig(
        N=int(N),
        H=int(H),
        W=int(W),
        C=int(C),
        K=int(K),
        R=int(R),
        S=int(S),
        array_size=int(array_size),
        stride=int(stride),
        pad=int(pad),
        seed=int(seed),
        use_sequential_init=use_sequential_init,
    )
    sim = ImplicitIm2colSystolicSim(cfg)
    unfolded = sim.explicit_im2col_channel_first()
    ofmap, logs = sim.simulate_systolic(trace=True, max_logs=int(max_logs))
    ref = direct_conv_hwc(sim.ifmap, sim.weights, stride=cfg.stride, pad=cfg.pad)
    diff = float(np.max(np.abs(ref - ofmap)))

    st.session_state["sim"] = sim
    st.session_state["unfolded"] = unfolded
    st.session_state["ofmap"] = ofmap
    st.session_state["logs"] = logs
    st.session_state["ref"] = ref
    st.session_state["diff"] = diff


if "sim" in st.session_state:
    sim = st.session_state["sim"]
    unfolded = st.session_state["unfolded"]
    ofmap = st.session_state["ofmap"]
    logs = st.session_state["logs"]
    ref = st.session_state["ref"]
    diff = st.session_state["diff"]

    tabs = st.tabs(["IFMap", "Weights", "Unfolded", "Iteration", "Output"])

    with tabs[0]:
        st.subheader("IFMap (HWC)")
        n_idx = st.number_input("Batch index", min_value=0, max_value=sim.cfg.N - 1, value=0, step=1)
        c_idx = st.number_input("Channel index", min_value=0, max_value=sim.cfg.C - 1, value=0, step=1)
        st.write(f"IFMap[n={n_idx}, :, :, c={c_idx}]")
        st.dataframe(_array_to_df(sim.ifmap[int(n_idx), :, :, int(c_idx)]))

    with tabs[1]:
        st.subheader("Weights (R,S,C,K)")
        r_idx = st.number_input("r index", min_value=0, max_value=sim.cfg.R - 1, value=0, step=1)
        s_idx = st.number_input("s index", min_value=0, max_value=sim.cfg.S - 1, value=0, step=1)
        st.write(f"Weights[r={r_idx}, s={s_idx}] (C x K)")
        st.dataframe(_array_to_df(sim.weights[int(r_idx), int(s_idx), :, :]))

    with tabs[2]:
        st.subheader("Unfolded IFMap (channel-first order)")
        st.dataframe(_array_to_df(unfolded))

    with tabs[3]:
        st.subheader("Iteration Viewer (systolic-timed, packed)")
        if not logs:
            st.info("No logs stored. Increase 'Max iterations to log'.")
        else:
            idx = st.slider("Iteration index", 0, len(logs) - 1, 0)
            entry = logs[int(idx)]
            st.write(f"Tiles: {entry['tile_rs_group']}, Out: {entry['out_hw']}")

            for t_idx, tile in enumerate(entry["tile_rs_group"]):
                st.markdown(f"**Tile {tile}**")
                st.write("Weight tile (C x K):")
                st.dataframe(_array_to_df(np.array(entry["weight_tiles"][t_idx])))
                st.write("Input word (C):")
                st.dataframe(_array_to_df(np.array(entry["input_words"][t_idx]).reshape(1, -1)))
                st.write("Partial sum (K):")
                st.dataframe(_array_to_df(np.array(entry["partial_sums"][t_idx]).reshape(1, -1)))

            st.write("Packed input (array_size):")
            st.dataframe(_array_to_df(np.array(entry["packed_input"]).reshape(1, -1)))
            st.write("Packed weight tile (array_size x K):")
            st.dataframe(_array_to_df(np.array(entry["packed_weight_tile"])))

            st.write(f"Output ready time: {entry['output_ready_time']}")
            timeline_rows = []
            for step in entry["timeline"]:
                timeline_rows.append(
                    {
                        "t": step["t"],
                        "row_inputs": step["row_inputs"],
                        "partial_sums": step["partial_sums"],
                    }
                )
            st.dataframe(_list_to_df(timeline_rows))

    with tabs[4]:
        st.subheader("Output")
        st.write("OFMap (HWC), channel 0:")
        st.dataframe(_array_to_df(ofmap[0, :, :, 0]))
        st.write("Direct conv reference, channel 0:")
        st.dataframe(_array_to_df(ref[0, :, :, 0]))
        st.write(f"Max abs diff: {diff}")
        st.write("PASS" if diff < 1e-6 else "FAIL")

else:
    st.info("Set parameters and click 'Run simulation'.")
