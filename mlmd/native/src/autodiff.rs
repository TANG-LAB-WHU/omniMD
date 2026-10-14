//! Minimal scalar reverse tape with differentiable directional tangents.
//! Tangent expressions are graph nodes, so reverse mode computes mixed derivatives.
use omnimd_ml_contract::{Error, Result};
pub(crate) type Id = usize;
#[derive(Clone, Copy)]
struct Node {
    value: f64,
    parents: [Id; 2],
    slopes: [f64; 2],
}
pub(crate) struct Tape {
    nodes: Vec<Node>,
    failed: bool,
}
#[derive(Clone, Copy)]
pub(crate) struct Dual {
    pub value: Id,
    pub tangent: Id,
}
impl Tape {
    pub fn new() -> Self {
        let mut tape = Self {
            nodes: Vec::new(),
            failed: false,
        };
        tape.leaf(0.0);
        tape.leaf(1.0);
        tape
    }
    fn push(&mut self, node: Node) -> Id {
        if self.nodes.len() >= 2_000_000 {
            self.failed = true;
            return 0;
        }
        if !node.value.is_finite() || node.slopes.iter().any(|x| !x.is_finite()) {
            self.failed = true;
        }
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }
    pub fn leaf(&mut self, value: f64) -> Id {
        self.push(Node {
            value,
            parents: [usize::MAX; 2],
            slopes: [0.0; 2],
        })
    }
    pub fn constant(&mut self, value: f64) -> Id {
        if value == 0.0 {
            0
        } else if value == 1.0 {
            1
        } else {
            self.leaf(value)
        }
    }
    pub fn value(&self, id: Id) -> f64 {
        self.nodes[id].value
    }
    pub fn add(&mut self, a: Id, b: Id) -> Id {
        if a == 0 {
            return b;
        }
        if b == 0 {
            return a;
        }
        self.push(Node {
            value: self.value(a) + self.value(b),
            parents: [a, b],
            slopes: [1.0, 1.0],
        })
    }
    pub fn mul(&mut self, a: Id, b: Id) -> Id {
        if a == 0 || b == 0 {
            return 0;
        }
        if a == 1 {
            return b;
        }
        if b == 1 {
            return a;
        }
        self.push(Node {
            value: self.value(a) * self.value(b),
            parents: [a, b],
            slopes: [self.value(b), self.value(a)],
        })
    }
    pub fn scale(&mut self, a: Id, k: f64) -> Id {
        let b = self.constant(k);
        self.mul(a, b)
    }
    fn tanh(&mut self, a: Id) -> Id {
        let y = self.value(a).tanh();
        self.push(Node {
            value: y,
            parents: [a, usize::MAX],
            slopes: [1.0 - y * y, 0.0],
        })
    }
    pub fn dual_constant(&mut self, value: f64, tangent: f64) -> Dual {
        Dual {
            value: self.constant(value),
            tangent: self.constant(tangent),
        }
    }
    pub fn dadd(&mut self, a: Dual, b: Dual) -> Dual {
        Dual {
            value: self.add(a.value, b.value),
            tangent: self.add(a.tangent, b.tangent),
        }
    }
    pub fn dmul(&mut self, a: Dual, b: Dual) -> Dual {
        let value = self.mul(a.value, b.value);
        let left = self.mul(a.tangent, b.value);
        let right = self.mul(a.value, b.tangent);
        Dual {
            value,
            tangent: self.add(left, right),
        }
    }
    pub fn dscale(&mut self, a: Dual, k: f64) -> Dual {
        Dual {
            value: self.scale(a.value, k),
            tangent: self.scale(a.tangent, k),
        }
    }
    pub fn dtanh(&mut self, a: Dual) -> Dual {
        let value = self.tanh(a.value);
        let square = self.mul(value, value);
        let negative = self.scale(square, -1.0);
        let slope = self.add(1, negative); // MUST remain differentiable, not a detached f64
        Dual {
            value,
            tangent: self.mul(slope, a.tangent),
        }
    }
    pub fn reverse(&self, seeds: &[(Id, f64)]) -> Result<Vec<f64>> {
        if self.failed {
            return Err(Error(
                "full-gradient tape exceeded two million nodes or became nonfinite".into(),
            ));
        }
        let mut adj = vec![0.0; self.nodes.len()];
        for &(id, value) in seeds {
            adj[id] += value;
        }
        for i in (0..self.nodes.len()).rev() {
            let n = self.nodes[i];
            for k in 0..2 {
                if n.parents[k] != usize::MAX {
                    adj[n.parents[k]] += adj[i] * n.slopes[k];
                }
            }
        }
        if adj.iter().any(|x| !x.is_finite()) {
            return Err(Error("nonfinite full-network adjoint".into()));
        }
        Ok(adj)
    }
}
