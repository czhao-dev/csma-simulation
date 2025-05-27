# IEEE 802.11 CSMA/CA Performance Analysis and Simulation

This repository contains an analytical and simulation-based exploration of the IEEE 802.11 CSMA/CA MAC protocol used in wireless networks. The project involves mathematical modeling, performance evaluation, and a discrete-event Python simulation to analyze system throughput, packet delay, and retransmissions under varying network loads.

## Overview

This project aims to evaluate and compare analytical and simulation-based performance metrics for a non-persistent CSMA/CA wireless system. The simulation closely follows the IEEE 802.11 protocol and compares its results against derived analytical expressions.

## Analytical Modeling

The analysis investigates a multiple-access wireless system where:
- N = 12 stations share a wireless channel
- Each station has a single-packet buffer
- Packet payload = 1800 bits + 240-bit header
- Data rate: 24 Mbps
- Slot time = 15 µs, DIFS = 60 µs, SIFS = 15 µs, Propagation delay = 2 µs

**Metrics derived analytically:**
- Probability of successful transmission in a slot
- Probability a station successfully transmits
- Throughput vs. Load (S vs. G)
- Maximum throughput `S_max` and corresponding `G*`
- Packet delay vs. throughput (D vs. S)
- Average number of retransmissions

## Equations and Derivations

Formulas were derived for:
- Successful transmission probabilities
- System throughput: `S(G)`
- Mean packet delay using binary exponential backoff
- Delay expectations as a function of retry limit and success probability

## Simulation

A Python-based discrete-event simulator emulates the CSMA/CA protocol operation, including:
- DIFS/SIFS interframe spacing
- Random backoff with exponential growth
- Packet acknowledgment (ACK) logic
- Packet arrival probability `q` per mini-slot

**Key metrics collected:**
- Throughput `S`
- Channel load `G`
- Head-of-Line (HOL) delay `D`
- Average number of retransmissions `E(N_T)`

**Simulation Time Frame:** 1000 mini-slots

## Performance Comparison

The simulation results are plotted and compared to the analytical model:
- `S vs. G`: Throughput vs. Load
- `D vs. S`: Packet delay vs. Throughput
- `E(N_T) vs. G`: Retransmissions vs. Load

## Files Included

- `csma_simulator.py`: Full Python source code for the CSMA/CA simulator
- Simulation output plots (generated dynamically)

## Observations

- Analytical and simulation results closely match, with minor deviations under saturation.
- Delay increases rapidly as throughput approaches maximum.
- Retransmission count grows significantly under high traffic load.

## Requirements

- Python 3.x
- `numpy`
- `matplotlib`

## Usage

To run the simulation and generate plots:

```bash
python csma_simulator.py
```

This will output:
- Throughput vs. Load (S vs. G)
- Delay vs. Throughput (D vs. S)
- Retransmissions vs. Load (E(N_T) vs. G)
- Maximum throughput value and corresponding channel load (G*)

## License
This project is released for academic and research purposes. Please credit the source if used in publications or derivative works.
