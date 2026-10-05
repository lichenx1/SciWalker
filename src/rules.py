REQUIRED_TOP_LEVEL_FIELDS = [
    "problem_name",
    "problem_id",
    "problem_description_main",
    "problem_background_main",
    "problem_io",
    "required_dependencies",
    "sub_steps",
    "general_solution",
    "general_tests",
    "oracle_dependencies",
    "oracle_solution",
    "oracle_tests",
    "problem_answer_main",
]

ALLOWED_IMPORT_HINT = (
    "Student-visible `required_dependencies` may use only general-purpose standard "
    "or SciCode-like base libraries: numpy as np; scipy modules such as integrate, "
    "optimize, signal, special, fft, linalg, sparse, interpolate, constants, spatial; "
    "math/cmath/itertools/collections/functools/copy/random; and sympy as sp. "
    "Do not import domain-specialized libraries such as Astropy, QuTiP, Qiskit, RDKit, "
    "OpenMM, pymatgen, PlasmaPy, SunPy, healpy, pyhf, or any hidden operator library. "
    "Do not directly import a single high-level solver/operator function that would solve "
    "a sub-step in one call; import modules and implement the algorithm from lower-level "
    "array, scalar, integration, optimization, interpolation, sparse, or symbolic building blocks."
)

FORBIDDEN_VISIBLE_TERMS = [
    "operator chain",
    "operator ids",
    "qutip",
    "cirq",
    "qiskit",
    "astropy",
    "plasmapy",
    "sunpy",
    "healpy",
    "pyscf",
    "rdkit",
    "openmm",
    "pymatgen",
    "kwant",
    "einsteinpy",
    "correlation_3op_1t",
    "wrightomega",
    "pexquo",
    "is_increasing",
    "epath",
    "msubs",
    "parallel_poly_from_expr",
    "count_roots",
    "int_valued",
    "idstn",
    "bravyi",
    "bravyi-maslov",
    "placeholder",
    "student must implement",
    "you may return",
]

BARE_CONSTRUCTOR_OUTPUT_PATTERNS = [
    "identity_op",
    "identity operator",
    "identity matrix",
    "zero matrix",
    "zero-initialized",
    "zero initialized",
    "rho_zero",
    "fock state vector",
    "basis vector",
    "state_vector",
    "state vector",
    "initial density matrix",
    "is_normalized",
    "normalization flag",
    "boolean flag indicating whether",
]

DIRECT_LIBRARY_API_VISIBLE_PATTERNS = [
    r"\b(?:use|using|call|calling|apply|applying|evaluate|evaluating|compute|computed|implement|implemented)\b[^.\n]{0,120}\b(?:np|numpy|scipy|sympy|sp)\s*\.",
    r"\b(?:scipy|sympy|numpy)\s*\.[A-Za-z_]\w+\b",
    r"\b(?:scipy|sympy|numpy)\s*\.[A-Za-z_]\w+(?:\s*\.[A-Za-z_]\w+)+",
    r"\b(?:np|sp)\s*\.[A-Za-z_]\w+\b",
]

ALLOWED_VISIBLE_LIBRARY_REFERENCES = {
    "np.ndarray",
    "numpy.ndarray",
}

NO_STUDENT_INPUT_PATTERNS = [
    r"^\s*:?\s*(?:none|no inputs?|no parameters?)\b",
    r"\ball parameters (?:are )?fixed\b",
    r"\bno external inputs?\b",
]

ALLOWED_GENERIC_OPERATOR_WORDS = {
    "collect",
    "factor",
    "integer",
    "degree",
    "det",
    "dct",
    "differential_entropy",
    "entropy",
    "fft",
    "fourier_transform",
    "limit",
    "matrix_rank",
    "mode",
    "momentum",
    "norm",
    "rank",
    "root",
    "resample",
    "series",
    "simplify",
    "solve",
    "spectrum",
    "svd",
    "sum",
    "tunneling",
    "variance",
    "wigner",
}

MIN_ANSWER_CHARS = 300

MIN_ANSWER_SENTENCES = 3

MIN_SUBSTEP_TEST_CASES = 3

MAX_SUBSTEP_TEST_CASES = 7

MIN_GENERAL_TESTS = 4

MAX_GENERAL_TESTS = 7

MIN_GENERAL_ASSERTIONS = 16

MIN_NONTRIVIAL_GENERAL_TESTS = 4

MIN_ORACLE_TESTS = 4

MIN_ORACLE_ASSERTIONS = 8

MIN_STUDENT_CODE_LINES = 35

MIN_STUDENT_FUNCTIONS = 3

MAX_RAISE_STATEMENTS = 12

MAX_RAISE_PER_CODE_LINE = 0.15

QUALITY_DIMENSION_MIN_SCORES = {
    "problem_difficulty_score_1_to_10": 6.0,
    "student_code_complexity_score_1_to_10": 6.0,
    "student_code_length_adequacy_score_1_to_10": 6.0,
    "test_strength_score_1_to_10": 7.0,
}

ALLOWED_REQUIRED_IMPORT_TOP_LEVEL = {
    "collections",
    "cmath",
    "copy",
    "functools",
    "itertools",
    "math",
    "numpy",
    "random",
    "scipy",
    "sympy",
}

ALLOWED_REQUIRED_IMPORT_MODULES = {
    "collections",
    "collections.abc",
    "cmath",
    "copy",
    "functools",
    "itertools",
    "math",
    "numpy",
    "numpy.fft",
    "numpy.linalg",
    "random",
    "scipy",
    "scipy.constants",
    "scipy.fft",
    "scipy.integrate",
    "scipy.interpolate",
    "scipy.linalg",
    "scipy.optimize",
    "scipy.signal",
    "scipy.sparse",
    "scipy.sparse.linalg",
    "scipy.spatial",
    "scipy.spatial.distance",
    "scipy.special",
    "sympy",
}

ALLOWED_FROM_IMPORT_MODULE_NAMES = {
    "constants",
    "fft",
    "integrate",
    "interpolate",
    "linalg",
    "optimize",
    "signal",
    "sparse",
    "spatial",
    "special",
}

FORBIDDEN_STUDENT_DEPENDENCY_TOP_LEVEL = {
    "ase",
    "astropy",
    "bio",
    "biopython",
    "cirq",
    "einsteinpy",
    "gala",
    "galpy",
    "healpy",
    "kwant",
    "openfermion",
    "openmm",
    "plasmapy",
    "poliastro",
    "pymatgen",
    "pyscf",
    "qiskit",
    "qutip",
    "rdkit",
    "rebound",
    "sunpy",
    "yt",
}

QUALITY_GRADE_ORDER = {"D": 0, "C": 1, "B": 2, "A": 3, "S": 4}

SA_RUBRIC_TEXT = """
S/A quality target:
- S means a polished, directly usable SciCode-style scientific problem: the domain
  background is natural, the final output is a meaningful scientific quantity,
  every input parameter affects the output or is explicitly justified, sub-steps
  form a real dependency chain, code implements the stated equations without
  hidden dummy parameters, and tests check nontrivial numeric values or domain
  invariants.
- A means high quality and directly usable with only minor simplification or
  wording issues; it still must be domain-coherent, parameterized, executable,
  non-template-like, and nontrivial enough to challenge a capable coding model.
- S/A requires real problem difficulty. The task should require multiple
  dependent reasoning or implementation steps, not just applying one formula,
  one standard library routine, or a short filter/aggregate. Student code length
  and helper structure should reflect scientific computation rather than
  validation boilerplate.
- S/A requires strong tests. General tests and oracle tests should detect wrong
  algorithms, wrong parameter sensitivity, wrong edge-case handling, and wrong
  numerical values, not just shape/type/field presence.
- Passing execution tests is not enough for A/S. If the task is mostly a thin
  wrapper around a standard decomposition/filter/rank/statistic, or if the final
  output is only a rank, boolean, count, p-value, central-interval proportion, or
  image shift without a rich scientific diagnostic, grade it B unless the domain
  story and tests are exceptionally strong.
- B means usable but not high quality: executable, coherent enough, but generic,
  stitched together from unrelated numerical operations, too template-like, too
  thin in final observable, or mostly numerical linear algebra dressed as a domain story.
- B/C/D are not acceptable for this run. Avoid common B/C/D failure modes:
  overused two-qubit/spectroscopy/templates from any domain, final outputs that are only
  constructors or booleans, visible API hints, unused inputs, domain stories not
  reflected in code, circular inference where true parameters are used to recover
  themselves, heavy sub-step tests that timeout, and tests that only check shape.
""".strip()

QUALITY_FEWSHOT_SUMMARY = """
Compressed quality examples:
- S example pattern: a natural domain measurement or modeling task where each input
  changes a named observable, sub-steps compute real intermediate quantities,
  student code needs substantial algorithmic work, and tests verify numerical
  values plus conservation/normalization/symmetry.
- A example pattern: a coherent SciCode-style scientific computation with minor
  simplification, clear IO, executable solution, several meaningful helper
  functions, and at least one parameter sensitivity or invariant test.
- B example pattern: executable and mostly coherent, but the story is generic,
  the student solution is short/thin, tests are not strong enough, one input is
  weakly used, or the sub-step chain feels somewhat artificial. Keep it, but do
  not call it high quality.
- B example pattern: a problem asks for a rank, interval proportion, matrix
  decomposition, filtered image, shifted phase-space map, or generic modal
  eigenvalues, and the domain context is plausible but not the real reason the
  computation is needed.
- C example pattern: visible API leakage, decorative domain story, constructor
  or boolean final output, unused/dummy parameters, weak shape-only tests, or
  sub-step timeouts. Keep it only for analysis/repair.
- D example pattern: not usable as a SciCode problem: broken semantics, circular
  inference, non-executable code after repair, severe leakage, or no meaningful
  scientific output.
""".strip()

FINAL_OUTPUT_DOMAIN_TERMS = [
    "accuracy",
    "spectrum",
    "response",
    "curve",
    "energy",
    "rate",
    "probability",
    "amplitude",
    "frequency",
    "mode",
    "observable",
    "density",
    "flux",
    "field",
    "map",
    "coefficient",
    "parameter",
    "estimate",
    "profile",
    "correlation",
    "power",
    "phase",
    "intensity",
    "diagnostic",
    "distance",
    "score",
    "risk",
    "return",
    "portfolio",
    "estimate",
    "confidence",
    "likelihood",
    "posterior",
    "classification",
    "alignment",
    "sequence",
    "mutation",
    "concentration",
    "yield",
    "mesh",
    "area",
    "volume",
    "curvature",
    "orbit",
    "trajectory",
    "structure",
    "property",
    "prediction",
]

TEMPLATE_PREFIX_PATTERNS = [
    r"^(?:two|three|single)_qubit",
    r"^quantum_(?:density|state|channel|spin)",
    r"^spectroscopy_(?:peak|spectrum|line)",
    r"^thermal_(?:diffusion|rod|material)",
    r"^laser_(?:beam|profile)",
    r"^diffusion_(?:profile|coefficient)",
]

PHYSICS_KEYWORDS = [
    "acoustic",
    "algorithm",
    "amplitude",
    "analysis",
    "antenna",
    "assay",
    "beam",
    "basis-state",
    "bio",
    "bond",
    "calibration",
    "catalysis",
    "cell",
    "chemical",
    "chemistry",
    "confidence",
    "coordinate",
    "covariance",
    "crystal",
    "dataset",
    "energy",
    "electric",
    "enzyme",
    "estimate",
    "finance",
    "potential",
    "pressure",
    "probability",
    "propagation",
    "propagator",
    "quantum",
    "qubit",
    "radiation",
    "wave",
    "field",
    "frequency",
    "spectrum",
    "scattering",
    "diffusion",
    "distance",
    "dna",
    "eigen",
    "oscillator",
    "dispersion",
    "fitness",
    "geometry",
    "graph",
    "lattice",
    "likelihood",
    "mesh",
    "molecule",
    "material",
    "matrix",
    "mutation",
    "orbit",
    "optimization",
    "thermal",
    "charge",
    "mode",
    "network",
    "portfolio",
    "protein",
    "reaction",
    "regression",
    "resonance",
    "risk",
    "sequence",
    "simulation",
    "statistic",
    "trajectory",
    "trajectory",
    "phase",
    "flux",
    "density",
    "magnetic",
    "optical",
    "polarization",
    "sensor",
    "signal",
    "structure",
    "variance",
]

SUBDOMAIN_BACKGROUND_CATALOG: dict[str, list[dict[str, str]]] = {
    "astrophysics": [
        {
            "id": "stellar_light_curve",
            "title": "stellar light-curve diagnostics",
            "description": "Analyze periodic flux, transit depth, flare statistics, or uncertainty-weighted variability.",
        },
        {
            "id": "galaxy_spectrum_summary",
            "title": "galaxy spectrum summary",
            "description": "Estimate redshift-like shifts, emission-line strengths, continuum trends, or spectral quality metrics.",
        },
        {
            "id": "sky_map_crossmatch",
            "title": "sky-map source crossmatch",
            "description": "Combine angular positions, photometric measurements, and catalog errors into ranked source diagnostics.",
        },
    ],
    "biochemistry": [
        {
            "id": "enzyme_kinetic_assay",
            "title": "enzyme kinetic assay",
            "description": "Estimate reaction rates, inhibition effects, Michaelis-like parameters, or replicate consistency.",
        },
        {
            "id": "protein_binding_curve",
            "title": "protein binding curve",
            "description": "Fit concentration-response data, binding fractions, cooperativity summaries, or uncertainty bands.",
        },
        {
            "id": "metabolite_panel_qc",
            "title": "metabolite panel quality control",
            "description": "Normalize assay panels, detect batch drift, and compute pathway-level summary scores.",
        },
    ],
    "bioinformatics_sequence_analysis": [
        {
            "id": "sequence_motif_enrichment",
            "title": "sequence motif enrichment",
            "description": "Scan sequences for motifs, background rates, positional enrichment, and calibrated scores.",
        },
        {
            "id": "alignment_quality_profile",
            "title": "alignment quality profile",
            "description": "Compute mismatch profiles, consensus summaries, gap penalties, and per-window reliability.",
        },
        {
            "id": "kmer_signature_distance",
            "title": "k-mer signature distance",
            "description": "Compare sequence collections using k-mer frequencies, smoothing, and distance-based diagnostics.",
        },
    ],
    "celestial_mechanics_orbital_dynamics": [
        {
            "id": "orbit_fit_residuals",
            "title": "orbit-fit residual diagnostics",
            "description": "Propagate orbital states, compare observations, and summarize residual or stability metrics.",
        },
        {
            "id": "transfer_window_scoring",
            "title": "transfer-window scoring",
            "description": "Evaluate maneuver timing, velocity budgets, phase angles, and feasibility margins.",
        },
        {
            "id": "resonance_stability_map",
            "title": "resonance stability map",
            "description": "Analyze period ratios, perturbation indicators, and boundedness over deterministic samples.",
        },
    ],
    "cheminformatics": [
        {
            "id": "molecular_descriptor_screen",
            "title": "molecular descriptor screen",
            "description": "Compute descriptor transforms, similarity scores, threshold sweeps, and candidate rankings.",
        },
        {
            "id": "fingerprint_calibration",
            "title": "fingerprint calibration",
            "description": "Analyze bit-count statistics, Tanimoto-like similarities, and calibration curves.",
        },
        {
            "id": "property_prediction_error",
            "title": "property prediction error",
            "description": "Aggregate molecular features, uncertainty weights, and regression diagnostics.",
        },
    ],
    "computational_chemistry": [
        {
            "id": "reaction_coordinate_profile",
            "title": "reaction coordinate profile",
            "description": "Compute energy profiles, barrier estimates, smoothed gradients, and confidence intervals.",
        },
        {
            "id": "vibrational_mode_summary",
            "title": "vibrational mode summary",
            "description": "Analyze Hessian-like matrices, mode frequencies, participation, and spectral broadening.",
        },
        {
            "id": "charge_distribution_qc",
            "title": "charge distribution quality control",
            "description": "Summarize partial charges, multipole-like moments, and geometry-weighted consistency checks.",
        },
    ],
    "computational_finance": [
        {
            "id": "portfolio_risk_attribution",
            "title": "portfolio risk attribution",
            "description": "Compute covariance perturbations, exposure contributions, stress scenarios, and risk summaries.",
        },
        {
            "id": "option_surface_calibration",
            "title": "option surface calibration",
            "description": "Fit implied-surface proxies, arbitrage-like diagnostics, and interpolation residuals.",
        },
        {
            "id": "factor_model_diagnostics",
            "title": "factor model diagnostics",
            "description": "Estimate factor loadings, residual covariance, explained variance, and scenario sensitivity.",
        },
    ],
    "computational_geometry_mesh_algorithms": [
        {
            "id": "mesh_quality_metrics",
            "title": "mesh quality metrics",
            "description": "Measure element shape, adjacency consistency, boundary defects, and refinement indicators.",
        },
        {
            "id": "point_cloud_registration",
            "title": "point-cloud registration",
            "description": "Compute geometric transforms, nearest-neighbor residuals, and robust alignment scores.",
        },
        {
            "id": "surface_curvature_summary",
            "title": "surface curvature summary",
            "description": "Estimate local normals, curvature-like diagnostics, and spatial smoothing reliability.",
        },
    ],
    "computational_mechanics": [
        {
            "id": "finite_element_response",
            "title": "finite-element response diagnostics",
            "description": "Assemble stiffness-like systems, solve displacements, and compute stress or compliance summaries.",
        },
        {
            "id": "vibration_mode_filtering",
            "title": "vibration mode filtering",
            "description": "Analyze modal frequencies, damping responses, and sensor-weighted mode contributions.",
        },
        {
            "id": "material_failure_margin",
            "title": "material failure margin",
            "description": "Combine load paths, strain proxies, and safety-factor maps into deterministic checks.",
        },
    ],
    "computational_physics": [
        {
            "id": "simulation_observable_summary",
            "title": "simulation observable summary",
            "description": "Compute time-series observables, conservation diagnostics, and parameter-sensitivity checks.",
        },
        {
            "id": "lattice_field_estimator",
            "title": "lattice field estimator",
            "description": "Aggregate grid fields, finite-difference operators, and energy or flux-like quantities.",
        },
        {
            "id": "stochastic_process_diagnostics",
            "title": "stochastic process diagnostics",
            "description": "Analyze deterministic seeded simulations, moments, correlations, and convergence indicators.",
        },
    ],
    "computational_statistics": [
        {
            "id": "bootstrap_interval_diagnostics",
            "title": "bootstrap interval diagnostics",
            "description": "Estimate resampled summaries, interval stability, bias corrections, and coverage indicators.",
        },
        {
            "id": "robust_regression_qc",
            "title": "robust regression quality control",
            "description": "Compute residual weights, leverage diagnostics, and sensitivity to outliers.",
        },
        {
            "id": "distribution_fit_comparison",
            "title": "distribution-fit comparison",
            "description": "Compare likelihood scores, calibrated residuals, and small-sample goodness diagnostics.",
        },
    ],
    "condensed_matter_physics": [
        {
            "id": "band_structure_summary",
            "title": "band-structure summary",
            "description": "Analyze dispersion grids, band gaps, density proxies, and finite-size trends.",
        },
        {
            "id": "spin_correlation_profile",
            "title": "spin correlation profile",
            "description": "Compute finite-chain correlations, structure factors, and susceptibility-like summaries.",
        },
        {
            "id": "transport_coefficient_estimate",
            "title": "transport coefficient estimate",
            "description": "Estimate conductance-like responses, disorder sensitivity, and scaling diagnostics.",
        },
    ],
    "crystallography_diffraction": [
        {
            "id": "diffraction_peak_refinement",
            "title": "diffraction peak refinement",
            "description": "Fit peak positions, lattice parameters, intensities, and residual quality metrics.",
        },
        {
            "id": "reciprocal_lattice_indexing",
            "title": "reciprocal-lattice indexing",
            "description": "Map reflections, reciprocal vectors, and consistency scores for indexed peaks.",
        },
        {
            "id": "structure_factor_summary",
            "title": "structure-factor summary",
            "description": "Compute amplitude summaries, phase-like diagnostics, and symmetry consistency checks.",
        },
    ],
    "ecology": [
        {
            "id": "population_stability_analysis",
            "title": "population stability analysis",
            "description": "Model population trajectories, equilibria, extinction risk, and intervention sensitivity.",
        },
        {
            "id": "species_interaction_network",
            "title": "species interaction network",
            "description": "Analyze interaction matrices, diversity indices, resilience, and perturbation effects.",
        },
        {
            "id": "habitat_suitability_curve",
            "title": "habitat suitability curve",
            "description": "Combine environmental covariates, thresholds, and calibrated suitability or risk scores.",
        },
    ],
    "fluid_dynamics": [
        {
            "id": "velocity_field_diagnostics",
            "title": "velocity-field diagnostics",
            "description": "Compute divergence, vorticity, flux, and energy summaries from gridded flow data.",
        },
        {
            "id": "boundary_layer_profile",
            "title": "boundary-layer profile",
            "description": "Analyze shear proxies, profile fits, and nondimensional response metrics.",
        },
        {
            "id": "reduced_order_flow_model",
            "title": "reduced-order flow model",
            "description": "Extract modal coefficients, reconstruction errors, and time-evolution diagnostics.",
        },
    ],
    "genetics": [
        {
            "id": "variant_frequency_panel",
            "title": "variant frequency panel",
            "description": "Compute allele frequencies, Hardy-like diagnostics, stratified summaries, and confidence bands.",
        },
        {
            "id": "genotype_trait_score",
            "title": "genotype-trait score",
            "description": "Aggregate variant weights, phenotype residuals, and calibration diagnostics.",
        },
        {
            "id": "linkage_window_summary",
            "title": "linkage-window summary",
            "description": "Analyze haplotype windows, co-occurrence patterns, and recombination-like indicators.",
        },
    ],
    "materials_informatics_structure_property_modeling": [
        {
            "id": "structure_property_surrogate",
            "title": "structure-property surrogate",
            "description": "Combine descriptors, regression residuals, uncertainty scores, and screening metrics.",
        },
        {
            "id": "defect_cluster_statistics",
            "title": "defect cluster statistics",
            "description": "Analyze defect positions, neighborhood summaries, strain proxies, and reliability scores.",
        },
        {
            "id": "composition_phase_screen",
            "title": "composition phase screen",
            "description": "Evaluate composition grids, stability proxies, and phase-boundary diagnostics.",
        },
    ],
    "molecular_modeling": [
        {
            "id": "conformation_energy_screen",
            "title": "conformation energy screen",
            "description": "Compare conformer energies, geometric descriptors, Boltzmann-like weights, and rank stability.",
        },
        {
            "id": "molecular_surface_summary",
            "title": "molecular surface summary",
            "description": "Compute surface descriptors, accessibility proxies, and parameter-sweep diagnostics.",
        },
        {
            "id": "trajectory_contact_map",
            "title": "trajectory contact map",
            "description": "Analyze molecular trajectories, contact frequencies, distance summaries, and clustering stability.",
        },
    ],
    "neuroscience_biophysics": [
        {
            "id": "neural_signal_filtering",
            "title": "neural signal filtering",
            "description": "Analyze spikes or field potentials, spectral bands, coherence, and event-triggered summaries.",
        },
        {
            "id": "membrane_response_curve",
            "title": "membrane response curve",
            "description": "Compute membrane-like responses, time constants, stimulus sensitivity, and stability checks.",
        },
        {
            "id": "connectivity_matrix_diagnostics",
            "title": "connectivity matrix diagnostics",
            "description": "Summarize network weights, centrality, synchrony proxies, and perturbation robustness.",
        },
    ],
    "nonlinear_dynamics_chaos": [
        {
            "id": "attractor_stability_map",
            "title": "attractor stability map",
            "description": "Compute trajectory summaries, sensitivity indicators, and parameter-dependent stability regions.",
        },
        {
            "id": "bifurcation_proxy_curve",
            "title": "bifurcation proxy curve",
            "description": "Analyze fixed points, local slopes, oscillation amplitudes, and regime transitions.",
        },
        {
            "id": "chaotic_signal_embedding",
            "title": "chaotic signal embedding",
            "description": "Estimate delay embeddings, recurrence summaries, and predictability diagnostics.",
        },
    ],
    "numerical_linear_algebra": [
        {
            "id": "matrix_conditioning_diagnostics",
            "title": "matrix conditioning diagnostics",
            "description": "Compute conditioning proxies, perturbation sensitivity, and solver residual summaries.",
        },
        {
            "id": "low_rank_model_error",
            "title": "low-rank model error",
            "description": "Analyze singular spectra, reconstruction errors, and cross-validated rank choices.",
        },
        {
            "id": "eigenvalue_sensitivity",
            "title": "eigenvalue sensitivity",
            "description": "Estimate eigenvalue shifts, modal participation, and residual-based reliability.",
        },
    ],
    "numerical_optimization": [
        {
            "id": "constrained_fit_diagnostics",
            "title": "constrained fit diagnostics",
            "description": "Solve parameter fits, constraint residuals, sensitivity curves, and convergence summaries.",
        },
        {
            "id": "calibration_surface_search",
            "title": "calibration surface search",
            "description": "Explore objective landscapes, local minima, regularization effects, and validation errors.",
        },
        {
            "id": "robust_parameter_estimation",
            "title": "robust parameter estimation",
            "description": "Combine loss functions, weights, bounds, and uncertainty estimates for fitted parameters.",
        },
    ],
    "optics": [
        {
            "id": "thin_film_response",
            "title": "thin-film response",
            "description": "Compute reflectance trends, phase response, polarization summaries, and spectral diagnostics.",
        },
        {
            "id": "beam_profile_quality",
            "title": "beam profile quality",
            "description": "Analyze intensity profiles, aperture effects, centroid shifts, and width estimates.",
        },
        {
            "id": "interferogram_phase_recovery",
            "title": "interferogram phase recovery",
            "description": "Estimate fringe phase, contrast, unwrapped trends, and noise sensitivity.",
        },
    ],
    "particle_physics": [
        {
            "id": "event_selection_efficiency",
            "title": "event selection efficiency",
            "description": "Compute selection efficiencies, background rates, significance proxies, and uncertainty bands.",
        },
        {
            "id": "invariant_mass_window",
            "title": "invariant-mass window",
            "description": "Analyze candidate masses, resolution effects, sideband summaries, and calibrated yields.",
        },
        {
            "id": "detector_response_unfolding",
            "title": "detector response unfolding",
            "description": "Estimate response corrections, regularized spectra, and closure-test diagnostics.",
        },
    ],
    "plasma_physics": [
        {
            "id": "plasma_wave_spectrum",
            "title": "plasma wave spectrum",
            "description": "Compute dispersion summaries, density fluctuation spectra, and diagnostic residuals.",
        },
        {
            "id": "particle_confinement_metric",
            "title": "particle confinement metric",
            "description": "Analyze trajectories, drift proxies, stability windows, and loss indicators.",
        },
        {
            "id": "probe_signal_calibration",
            "title": "probe signal calibration",
            "description": "Calibrate probe measurements, infer plasma parameters, and quantify uncertainty.",
        },
    ],
    "quantum_chemistry": [
        {
            "id": "orbital_energy_diagnostics",
            "title": "orbital energy diagnostics",
            "description": "Analyze orbital-like spectra, occupation summaries, perturbation shifts, and convergence metrics.",
        },
        {
            "id": "electron_density_projection",
            "title": "electron density projection",
            "description": "Compute density summaries, basis projections, charge consistency, and spatial moments.",
        },
        {
            "id": "small_molecule_correlation",
            "title": "small-molecule correlation estimate",
            "description": "Estimate correlation-like corrections, active-space summaries, and energy sensitivity.",
        },
    ],
    "quantum_information_quantum_computing": [
        {
            "id": "quantum_state_diagnostics",
            "title": "quantum state diagnostics",
            "description": "Compute state fidelities, reduced-state summaries, entropies, and measurement probabilities.",
        },
        {
            "id": "channel_noise_summary",
            "title": "channel noise summary",
            "description": "Analyze channel matrices, error rates, distinguishability, and small-system benchmarks.",
        },
        {
            "id": "circuit_observable_estimate",
            "title": "circuit observable estimate",
            "description": "Estimate observables from gate-like transformations, sampling probabilities, and sensitivity checks.",
        },
    ],
    "reaction_kinetics_catalysis": [
        {
            "id": "kinetic_rate_fit",
            "title": "kinetic rate fit",
            "description": "Estimate rate constants, activation trends, residual diagnostics, and confidence intervals.",
        },
        {
            "id": "catalyst_activity_screen",
            "title": "catalyst activity screen",
            "description": "Combine conversion data, selectivity, temperature effects, and ranking stability.",
        },
        {
            "id": "reaction_network_flux",
            "title": "reaction network flux",
            "description": "Analyze coupled reaction rates, steady states, sensitivity, and mass-balance checks.",
        },
    ],
    "relativity_gravitation": [
        {
            "id": "lensing_observable_fit",
            "title": "lensing observable fit",
            "description": "Compute magnification-like summaries, image residuals, caustic proxies, and uncertainty.",
        },
        {
            "id": "orbital_precession_signal",
            "title": "orbital precession signal",
            "description": "Analyze trajectory shifts, relativistic corrections, and residual timing diagnostics.",
        },
        {
            "id": "metric_curve_diagnostic",
            "title": "metric curve diagnostic",
            "description": "Evaluate coordinate curves, geodesic-like summaries, and invariant consistency checks.",
        },
    ],
    "semiconductor_materials": [
        {
            "id": "carrier_transport_curve",
            "title": "carrier transport curve",
            "description": "Compute mobility proxies, carrier density trends, bias response, and uncertainty.",
        },
        {
            "id": "bandgap_screening_metric",
            "title": "bandgap screening metric",
            "description": "Analyze composition descriptors, gap estimates, defect penalties, and rank stability.",
        },
        {
            "id": "device_response_calibration",
            "title": "device response calibration",
            "description": "Calibrate current-voltage-like curves, threshold metrics, and sensitivity to noise.",
        },
    ],
    "statistical_physics": [
        {
            "id": "ensemble_observable_curve",
            "title": "ensemble observable curve",
            "description": "Compute ensemble averages, susceptibilities, heat-like responses, and finite-size trends.",
        },
        {
            "id": "markov_state_diagnostics",
            "title": "Markov state diagnostics",
            "description": "Analyze transition matrices, stationary summaries, autocorrelations, and mixing indicators.",
        },
        {
            "id": "spin_model_summary",
            "title": "spin-model summary",
            "description": "Compute magnetization-like moments, correlations, and temperature-dependent response curves.",
        },
    ],
    "systems_biology": [
        {
            "id": "gene_network_response",
            "title": "gene network response",
            "description": "Analyze network dynamics, perturbation responses, pathway scores, and stability metrics.",
        },
        {
            "id": "signaling_pathway_flux",
            "title": "signaling pathway flux",
            "description": "Compute pathway flows, feedback sensitivity, steady-state summaries, and intervention effects.",
        },
        {
            "id": "population_cell_state_model",
            "title": "population cell-state model",
            "description": "Model cell-state proportions, transition rates, and trajectory consistency diagnostics.",
        },
    ],
}

NONTRIVIAL_GENERAL_TEST_PATTERNS = [
    r"\bnp\.(?:allclose|isclose|all|any|isfinite)\b",
    r"\bnumpy\.testing\.",
    r"\bassert_allclose\b",
    r"\babs\s*\(",
    r"\b(?:sum|mean|std|var|norm|trace|det|eigvalsh|trapz|dot)\s*\(",
    r"\bnp\.(?:sum|mean|std|var|linalg\.norm|trace|trapz|dot|vdot)\b",
    r"(?:<=|>=|<|>)",
    r"\b(?:conserv|normaliz|positive|nonnegative|monotonic|bounded|symmetric|hermitian)\b",
]

SHALLOW_GENERAL_TEST_PATTERNS = [
    r"\.shape\b",
    r"\bdtype\b",
    r"\bisinstance\s*\(",
    r"\btype\s*\(",
    r"\blen\s*\(",
]
