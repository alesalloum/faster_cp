mod algorithms;
mod graph;

use algorithms::simulated_annealing_partition;
use graph::GraphWrapper;
use petgraph::graph::UnGraph;
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyfunction]
fn add(a: f64, b: f64) -> PyResult<f64> {
    Ok(a + b)
}

#[pyfunction]
fn multiply(a: f64, b: f64) -> PyResult<f64> {
    Ok(a * b)
}

#[pyfunction]
fn build_graph(edges: Vec<(i32, i32)>) -> PyResult<usize> {
    let mut graph = UnGraph::<(), ()>::new_undirected();

    let mut node_map = std::collections::HashMap::new();

    for (u, v) in edges {
        let u_index = *node_map.entry(u).or_insert_with(|| graph.add_node(()));
        let v_index = *node_map.entry(v).or_insert_with(|| graph.add_node(()));

        graph.add_edge(u_index, v_index, ());
    }

    Ok(graph.node_count())
}

#[pymodule]
fn faster_cp(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m.clone())?)?; // ✅ Use `.clone()`
    m.add_function(wrap_pyfunction!(multiply, m.clone())?)?; // ✅ Register multiply
    m.add_function(wrap_pyfunction!(build_graph, m.clone())?)?;
    m.add_class::<GraphWrapper>()?;
    m.add_function(wrap_pyfunction!(simulated_annealing_partition, m.clone())?)?;
    Ok(())
}
