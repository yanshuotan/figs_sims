param_grids = {
    "linear": {
        "heritability": [0.6, 0.8, 0.9],
        "s": [2, 4, 6],
        "sigma": [0],
        "beta": [1]
    },
    "lss": {
        "heritability": [0.6, 0.8, 0.9],
        "m": [1, 3, 5],
        "r": [1, 2, 4],
        "sigma": [0],
        "tau": [0],
        "beta": [1]
    },
    "poly": {
        "heritability": [0.6, 0.8, 0.9],
        "m": [1, 2, 3],
        "r": [1, 2, 3],
        "sigma": [0],
        "beta": [1]
    },
    "hierarchical_lss": {
        "heritability": [0.6, 0.8, 0.9],
        "m": [1, 3, 5],
        "r": [1, 2, 4],
        "sigma": [0],
        "beta": [1],
        "lss": [True]
    }
}