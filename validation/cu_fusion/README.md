# Cu fusion check, 2026-10-08

Mendelev EAM Cu, 864 atoms, 1300 K, 121 NVT frames 0.5 ps apart (the 2026-10-02 red-team
trajectories in odinzen/tdb-assessment, entropy/binary_cuzr_eam/runs/fus_{solid,liquid}.dump).
Exact entropies from calphy on the same potential: solid 72.76, liquid 82.5, fusion 9.78 J/(K mol).

| Route | Solid | Liquid | Fusion |
|---|---|---|---|
| Hermes 0.1.0 (60 pairs, + 5.3 offset) | 77.8 | 98.8 | 21.1 (2.2x) |
| Hermes minus Gaussian nearest-neighbour pair correlation | 72.7-72.9 | 92.8-94.0 | about 20 |
| Ideal gas + two-body excess entropy from g(r) | n/a | 81.0 | |
| Exact (calphy) | 72.76 | 82.5 | 9.78 |

- Per-atom coding counts each atom as independent, so it gives the sum of marginal entropies. The
  solid excess (+5.0) equals the nearest-neighbour displacement mutual information (5.1), and
  subtracting it lands on the exact value.
- Zentropy over inherent structures does not rescue the liquid at 1300 K: quenching moves atoms
  0.92 A rms (solid 0.35), and the frame-to-basin compression alone (100.0) exceeds the total.
- Scripts expect the red-team dumps at the paths above and the inherent structures from quench.lmp
  (LAMMPS, WSL env lammps-full). Not packaged; one system, one temperature.
