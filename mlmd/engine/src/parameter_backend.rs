//! Engine-only adaptation of explicit parameter context to the official C API.
use super::BackendOptions;
use omnimd_deepmd::{DeepMd, Device, EvalParameters, EvaluationOptions, FrameParameterPolicy};
use omnimd_ml_contract::{Error, Evaluation, Frame, ModelMetadata, Potential, Result};
use omnimd_mlmd::parameters::{FixedParameters, FramePolicy, Layout, Values};

pub fn validate_request(options: &BackendOptions, layout: Option<&Layout>) -> Result<()> {
    if let Some(layout) = layout {
        layout.validate()?;
        if !matches!(options, BackendOptions::DeepMd(_)) || options.device() != Device::Cpu {
            return Err(Error("engine parameterized inference currently requires CPU DeepMD C API; native/CUDA parameter routes unsupported".into()));
        }
    }
    Ok(())
}
pub fn validate_layout(model: &DeepMd, layout: &Layout) -> Result<()> {
    layout.validate()?;
    let details = model.details();
    if layout.fparam.len() != details.frame_parameter_dim
        || layout.aparam.len() != details.atom_parameter_dim
    {
        return Err(Error(format!(
            "parameter layout dimensions ({},{}) do not match runtime model ({},{})",
            layout.fparam.len(),
            layout.aparam.len(),
            details.frame_parameter_dim,
            details.atom_parameter_dim
        )));
    }
    Ok(())
}
pub fn evaluate(
    model: &mut DeepMd,
    frame: &Frame,
    layout: &Layout,
    values: &Values,
) -> Result<Evaluation> {
    validate_layout(model, layout)?;
    values.validate(layout, frame.positions.len())?;
    let parameters = EvalParameters {
        fparam: values.fparam.clone(),
        aparam: values.aparam.clone(),
    };
    let policy = match values.frame_parameter_policy {
        FramePolicy::Explicit => FrameParameterPolicy::Explicit,
        FramePolicy::ModelDefault => FrameParameterPolicy::ModelDefault,
    };
    let mut result = model.evaluate_with_options(
        std::slice::from_ref(frame),
        Some(std::slice::from_ref(&parameters)),
        EvaluationOptions {
            atomic: false,
            frame_parameter_policy: policy,
        },
    )?;
    if result.len() != 1 {
        return Err(Error(
            "parameterized single-frame call returned wrong output count".into(),
        ));
    }
    Ok(result.remove(0).evaluation)
}
pub struct ConditionedPotential {
    pub model: DeepMd,
    pub fixed: FixedParameters,
}
impl Potential for ConditionedPotential {
    fn metadata(&self) -> &ModelMetadata {
        self.model.metadata()
    }
    fn eval(&mut self, frame: &Frame) -> Result<Evaluation> {
        self.fixed.validate(frame)?;
        evaluate(
            &mut self.model,
            frame,
            &self.fixed.layout,
            &self.fixed.values,
        )
    }
}
