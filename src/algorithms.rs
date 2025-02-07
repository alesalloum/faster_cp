use crate::graph::GraphWrapper;
use petgraph::graph::NodeIndex;
use pyo3::types::PyDictMethods;
use pyo3::PyObject;
use pyo3::{pyfunction, types::PyDict, PyResult, Python};
use rand::prelude::*;
use rand::seq::IteratorRandom;
use special::Gamma;
use std::{collections::HashMap, f64::INFINITY}; // For loggamma function

const PENALTY_VALUE: f64 = 1000.0;

fn relabel_partition(
    g: &GraphWrapper,
    best_partition: &HashMap<NodeIndex, i8>,
) -> HashMap<i32, i8> {
    // Reverse the `node_map` (Rust indices → original labels)
    let reversed_map: HashMap<NodeIndex, i32> = g.node_map.iter().map(|(&k, &v)| (v, k)).collect();

    // Create a new HashMap with relabeled nodes
    let mut relabeled_partition = HashMap::new();
    for (&key, &value) in best_partition.iter() {
        if let Some(&mapped_value) = reversed_map.get(&key) {
            relabeled_partition.insert(mapped_value, value); // Store relabeled partition
        }
    }

    relabeled_partition
}

fn compute_move_impact(
    g: &GraphWrapper,
    current_partition: &HashMap<NodeIndex, i8>,
    node: NodeIndex,
) -> (i32, i32, i32) {
    let orig_group = current_partition[&node]; // Get current group (0 or 1)
    let new_group = 1 - orig_group; // Flip group

    let mut delta_m00 = 0;
    let mut delta_m11 = 0;
    let mut delta_m01 = 0;

    // First, remove contributions of the selected node from current counts
    for neighbor in g.graph.neighbors(node) {
        if current_partition[&neighbor] == orig_group {
            if orig_group == 0 {
                delta_m00 -= 1;
            } else {
                delta_m11 -= 1;
            }
        } else {
            delta_m01 -= 1;
        }
    }

    // Next, add contributions when the node is assigned to the new group
    for neighbor in g.graph.neighbors(node) {
        if current_partition[&neighbor] == new_group {
            if new_group == 0 {
                delta_m00 += 1;
            } else {
                delta_m11 += 1;
            }
        } else {
            delta_m01 += 1;
        }
    }

    (delta_m00, delta_m11, delta_m01)
}

#[pyfunction]
pub fn simulated_annealing_partition(
    py: Python<'_>,
    g: &GraphWrapper,  //  Reference to GraphWrapper (graph + partition)
    max_iter: usize,   //  Max iterations (default: 10,000)
    initial_temp: f64, //  Initial temperature (default: 1.0)
    cooling_rate: f64, //  Cooling rate (default: 0.999)
) -> PyResult<PyObject> {
    // Placeholder logic (to be implemented)
    println!(
        "Running simulated annealing with max_iter={}, initial_temp={}, cooling_rate={}",
        max_iter, initial_temp, cooling_rate
    );

    // Initalize rng object
    let mut rng: ThreadRng = rand::rng();

    // Create copy of the initial partition
    let mut current_partition = g.partition.clone();

    // Initialize edge count variables (m00, m11, m01)
    let mut m00 = 0;
    let mut m11 = 0;
    let mut m01 = 0;

    // Count nodes in each group
    let mut n0 = 0;
    for (_node, &group) in &current_partition {
        if group == 0 {
            n0 += 1;
        }
    }
    let mut n1 = current_partition.len() - n0; // ✅ Compute n1 from total nodes

    // Count edges in each block
    for edge in g.graph.edge_indices() {
        let (u, v) = g.graph.edge_endpoints(edge).unwrap();

        // Get the partition group of each node
        let group_u = *current_partition
            .get(&u)
            .expect("Node not found in partition");
        let group_v = *current_partition
            .get(&v)
            .expect("Node not found in partition");

        if group_u == group_v {
            // Both nodes belong to the same group
            if group_u == 0 {
                m00 += 1; // Edge within group 0
            } else {
                m11 += 1; // Edge within group 1
            }
        } else {
            // Nodes belong to different groups
            m01 += 1;
        }
    }

    // Keep listaa on optimal partition
    let mut best_partition = current_partition.clone();

    // Compute the initial description length, which is "the best so far"
    let n = g.graph.node_count();
    let mut best_dl = compute_dl_from_counts(m00, m11, m01, n0, n1, n);
    let mut current_dl = best_dl;

    // Set starting temperature
    let mut T = initial_temp;

    /*
    // Define a random node generator
    fn select_random_node(g: &GraphWrapper, rng: &mut ThreadRng) -> NodeIndex {
        //let mut rng = thread_rng(); // Random number generator, this doesn't work!
        //let mut rng = rand::rng();
        let nodes: Vec<NodeIndex> = g.graph.node_indices().collect();
        nodes[rng.random_range(0..=nodes.len() - 1)] // Select a random node
    }
    */

    // Start the loop so to speak!
    for iter_idx in 1..=max_iter {
        //let random_node = select_random_node(&g, &mut rng);
        let random_node = g
            .graph
            .node_indices()
            .choose(&mut rng)
            .expect("Graph has no nodes!");
        let (delta_m00, delta_m11, delta_m01) =
            compute_move_impact(g, &current_partition, random_node);

        //println!("Iter nro: {}", iter_idx);
        //println!("Change in m00: {}", delta_m00);
        //println!("Change in m11: {}", delta_m11);
        //println!("Change in m01: {}", delta_m01);

        // Simply update the group sizes
        let new_n0 = if current_partition[&random_node] == 0 {
            n0 - 1
        } else {
            n0 + 1
        };
        let new_n1 = if current_partition[&random_node] == 0 {
            n1 + 1
        } else {
            n1 - 1
        };

        // Also update the edge counts based on the computed deltas (Check whether there's a smarter way to handle the conversion)
        let new_m00 = (m00 as i32 + delta_m00) as usize;
        let new_m11 = (m11 as i32 + delta_m11) as usize;
        let new_m01 = (m01 as i32 + delta_m01) as usize;

        let new_dl = compute_dl_from_counts(new_m00, new_m11, new_m01, new_n0, new_n1, n);
        let delta = new_dl - current_dl;

        // Write next logic for the approval of the partition candidate...
        if delta < 0.0 || rng.random_range(0.0..1.0) < (-delta / T).exp() {
            let orig_group = current_partition[&random_node];
            let new_group = 1 - orig_group;

            //Update the group for the node
            current_partition.insert(random_node, new_group);

            // Update also the "global" optima
            // ✅ Update global edge counts and group sizes
            m00 = new_m00;
            m11 = new_m11;
            m01 = new_m01;
            n0 = new_n0;
            n1 = new_n1;
            current_dl = new_dl;

            // New isnt necessarily the best, because of T
            if current_dl < best_dl {
                best_dl = current_dl;
                best_partition = current_partition.clone();
            }
        }

        // Temperature cooling
        T *= cooling_rate;

        // **Progress Output**
        if iter_idx % 10000 == 0 {
            println!(
                "Iteration {}, Best DL: {:.2}, Current DL: {:.2}, Temperature: {:.4}",
                iter_idx, best_dl, current_dl, T
            );
        }
    }

    // Debug print
    /*println!(
        "Initialized: n0 = {}, n1 = {}, edges = {}, ({}, {}, {}, {}), adjacency list built",
        n0,
        n1,
        g.graph.edge_count(),
        m00,
        m01,
        m11,
        best_dl
    );
    */

    // Prepare returning the best partition

    let relabeled_best_partition = relabel_partition(&g, &best_partition);

    // Create a Python dictionary and populate it with the partition data.
    // Assuming `relabeled_best_partition` is a HashMap where keys and values are convertible to Python objects.
    let py_dict = PyDict::new(py);
    for (key, value) in relabeled_best_partition.iter() {
        py_dict.set_item(*key, *value)?; // Convert NodeIndex → int
    }

    // Return the dictionary to Python.
    Ok(py_dict.into()) // Return as PyObject

    // Return Ok() to indicate success
}

fn compute_dl_from_counts(
    m00: usize,
    m11: usize,
    m01: usize,
    n0: usize,
    n1: usize,
    n: usize,
) -> f64 {
    // If one group is empty, return infinity (invalid partition)
    if n0 == 0 || n1 == 0 {
        return INFINITY;
    }

    // Compute the maximum possible number of edges in each group
    let M00 = if n0 > 1 { n0 * (n0 - 1) / 2 } else { 0 };
    let M11 = if n1 > 1 { n1 * (n1 - 1) / 2 } else { 0 };
    let M01 = n0 * n1;

    // Compute observed densities
    let dens0 = if M00 > 0 {
        m00 as f64 / M00 as f64
    } else {
        0.0
    };
    let dens1 = if M11 > 0 {
        m11 as f64 / M11 as f64
    } else {
        0.0
    };

    // Determine core-periphery structure
    let (p11_est, p22_est, m_core, M_core, m_peri, M_peri) = if dens0 >= dens1 {
        (dens0, dens1, m00, M00, m11, M11) // Group 0 is core
    } else {
        (dens1, dens0, m11, M11, m00, M00) // Group 1 is core
    };

    // Compute cross-group edge probability
    let p12_est = if M01 > 0 {
        m01 as f64 / M01 as f64
    } else {
        0.0
    };

    // Apply penalty if density conditions are violated
    let penalty = if !(p11_est > p12_est && p12_est > p22_est) {
        PENALTY_VALUE
    } else {
        0.0
    };

    // Define safe_log function
    fn safe_log(x: f64) -> f64 {
        if x > 0.0 && x < 1.0 {
            x.ln()
        } else {
            PENALTY_VALUE
        }
    }

    // Compute L_data (negative log-likelihood of the observed edges)
    let l_data = -(m_core as f64 * safe_log(p11_est)
        + (M_core - m_core) as f64 * safe_log(1.0 - p11_est)
        + m01 as f64 * safe_log(p12_est)
        + (M01 - m01) as f64 * safe_log(1.0 - p12_est)
        + m_peri as f64 * safe_log(p22_est)
        + (M_peri - m_peri) as f64 * safe_log(1.0 - p22_est));

    // Compute L_params (model parameter complexity penalty)
    let prior_cost: f64 = 6.0;
    let l_params = -prior_cost.ln(); // -math.log(6) in Python

    // Compute L_part (partition complexity)
    let (ln_gamma_val, _) = (n as f64 - 1.0).ln_gamma(); // Extract first value
    let l_part = -((n0 as f64).ln() + (n1 as f64).ln()
        - (n as f64).ln()
        - ((n - 1) as f64).ln()
        - ln_gamma_val); // Convert f32 to f64

    // Compute total description length
    let dl = l_data + l_params + l_part + penalty;

    // Return
    dl
}
