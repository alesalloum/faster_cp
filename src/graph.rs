use petgraph::graph::{NodeIndex, UnGraph};
use pyo3::{prelude::*, types::PyDict};
use std::collections::HashMap;

#[pyclass]
pub struct GraphWrapper {
    pub graph: UnGraph<(), ()>,
    pub node_map: HashMap<i32, NodeIndex>, // Mapping between Python IDs and Rust NodeIndex
    pub partition: HashMap<NodeIndex, i8>,
}

#[pymethods]
impl GraphWrapper {
    #[new]
    pub fn new() -> Self {
        GraphWrapper {
            graph: UnGraph::<(), ()>::new_undirected(),
            node_map: HashMap::new(),
            partition: HashMap::new(),
        }
    }

    pub fn add_edges(&mut self, py_edges: Vec<(i32, i32)>) {
        for (u, v) in py_edges {
            let u_index = *self
                .node_map
                .entry(u)
                .or_insert_with(|| self.graph.add_node(()));
            let v_index = *self
                .node_map
                .entry(v)
                .or_insert_with(|| self.graph.add_node(()));
            self.graph.add_edge(u_index, v_index, ());
        }
    }

    pub fn set_partition(&mut self, py_init_partition: &Bound<'_, PyDict>) -> PyResult<()> {
        for (key, value) in py_init_partition.into_iter() {
            let node_id: i32 = key.extract()?;
            let partition_id: i8 = value.extract()?;

            if let Some(&node_index) = self.node_map.get(&node_id) {
                self.partition.insert(node_index, partition_id);
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Node ID {} not found in the graph",
                    node_id
                )));
            }
        }
        Ok(())
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn count_partition_value(&self, value: i8) -> usize {
        self.partition.values().filter(|&&v| v == value).count()
    }
}
