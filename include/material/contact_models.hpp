#pragma once
#include "material/device_qualifier.cuh"

#include <memory>
#include <string>

namespace matgpu {

// Strategy pattern for contact models
class ContactModel {
  public:
    virtual ~ContactModel() = default;

    // Contact model identification
    virtual std::string model_name() const = 0;
    virtual uint8_t model_id() const       = 0;

    // Contact force computation interface
    virtual float compute_normal_force(float overlap, float velocity) const                           = 0;
    virtual float compute_tangential_force(float tangential_overlap, float tangential_velocity) const = 0;

    // Model-specific parameters access
    virtual bool has_parameter(const std::string& name) const        = 0;
    virtual float get_parameter(const std::string& name) const       = 0;
    virtual void set_parameter(const std::string& name, float value) = 0;

    // Device compatibility
    virtual bool is_device_compatible() const           = 0;
    virtual std::unique_ptr<ContactModel> clone() const = 0;
};

// EEPA (Elastic-Elastic-Plastic-Adhesive) contact model
class EEPAContactModel : public ContactModel {
  private:
    float kn_;      // Normal stiffness
    float kt_;      // Tangential stiffness
    float gamma_n_; // Normal damping
    float gamma_t_; // Tangential damping

  public:
    EEPAContactModel(float kn = 1e5f, float kt = 5e4f, float gamma_n = 0.2f, float gamma_t = 0.1f)
        : kn_(kn), kt_(kt), gamma_n_(gamma_n), gamma_t_(gamma_t) {}

    // ContactModel interface
    std::string model_name() const override { return "EEPA"; }
    uint8_t model_id() const override { return 1; }

    float compute_normal_force(float overlap, float velocity) const override {
        return kn_ * overlap + gamma_n_ * velocity;
    }

    float compute_tangential_force(float tangential_overlap, float tangential_velocity) const override {
        return kt_ * tangential_overlap + gamma_t_ * tangential_velocity;
    }

    bool has_parameter(const std::string& name) const override {
        return name == "kn" || name == "kt" || name == "gamma_n" || name == "gamma_t";
    }

    float get_parameter(const std::string& name) const override {
        if (name == "kn")
            return kn_;
        if (name == "kt")
            return kt_;
        if (name == "gamma_n")
            return gamma_n_;
        if (name == "gamma_t")
            return gamma_t_;
        return 0.0f;
    }

    void set_parameter(const std::string& name, float value) override {
        if (name == "kn")
            kn_ = value;
        else if (name == "kt")
            kt_ = value;
        else if (name == "gamma_n")
            gamma_n_ = value;
        else if (name == "gamma_t")
            gamma_t_ = value;
    }

    bool is_device_compatible() const override { return true; }

    std::unique_ptr<ContactModel> clone() const override {
        return std::make_unique<EEPAContactModel>(kn_, kt_, gamma_n_, gamma_t_);
    }

    // Direct parameter access for performance
    float kn() const { return kn_; }
    float kt() const { return kt_; }
    float gamma_n() const { return gamma_n_; }
    float gamma_t() const { return gamma_t_; }

    void set_stiffness(float kn, float kt) {
        kn_ = kn;
        kt_ = kt;
    }
    void set_damping(float gamma_n, float gamma_t) {
        gamma_n_ = gamma_n;
        gamma_t_ = gamma_t;
    }
};

// JKR (Johnson-Kendall-Roberts) contact model
class JKRContactModel : public ContactModel {
  private:
    float work_of_adhesion_; // J/m^2
    float contact_radius0_;  // Initial contact radius

  public:
    JKRContactModel(float work_of_adhesion = 0.05f, float contact_radius0 = 1e-4f)
        : work_of_adhesion_(work_of_adhesion), contact_radius0_(contact_radius0) {}

    // ContactModel interface
    std::string model_name() const override { return "JKR"; }
    uint8_t model_id() const override { return 2; }

    float compute_normal_force(float overlap, float velocity) const override {
        // Simplified JKR force calculation
        float radius = contact_radius0_ + overlap * 0.1f;    // Simplified
        return 6.0f * 3.14159f * work_of_adhesion_ * radius; // 6πγR
    }

    float compute_tangential_force(float tangential_overlap, float tangential_velocity) const override {
        // JKR typically uses friction-based tangential forces
        return 0.3f * compute_normal_force(0, 0) * tangential_overlap; // μ * Fn * displacement
    }

    bool has_parameter(const std::string& name) const override {
        return name == "work_of_adhesion" || name == "contact_radius0";
    }

    float get_parameter(const std::string& name) const override {
        if (name == "work_of_adhesion")
            return work_of_adhesion_;
        if (name == "contact_radius0")
            return contact_radius0_;
        return 0.0f;
    }

    void set_parameter(const std::string& name, float value) override {
        if (name == "work_of_adhesion")
            work_of_adhesion_ = value;
        else if (name == "contact_radius0")
            contact_radius0_ = value;
    }

    bool is_device_compatible() const override { return true; }

    std::unique_ptr<ContactModel> clone() const override {
        return std::make_unique<JKRContactModel>(work_of_adhesion_, contact_radius0_);
    }

    // Direct parameter access
    float work_of_adhesion() const { return work_of_adhesion_; }
    float contact_radius0() const { return contact_radius0_; }

    void set_adhesion_work(float work) { work_of_adhesion_ = work; }
    void set_initial_radius(float radius) { contact_radius0_ = radius; }
};

// No-contact model (elastic only)
class NoContactModel : public ContactModel {
  public:
    std::string model_name() const override { return "None"; }
    uint8_t model_id() const override { return 0; }

    float compute_normal_force(float, float) const override { return 0.0f; }
    float compute_tangential_force(float, float) const override { return 0.0f; }

    bool has_parameter(const std::string&) const override { return false; }
    float get_parameter(const std::string&) const override { return 0.0f; }
    void set_parameter(const std::string&, float) override {}

    bool is_device_compatible() const override { return true; }

    std::unique_ptr<ContactModel> clone() const override { return std::make_unique<NoContactModel>(); }
};

// Factory for contact models
class ContactModelFactory {
  public:
    static std::unique_ptr<ContactModel> create(const std::string& model_name) {
        if (model_name == "EEPA" || model_name == "eepa" || model_name == "HERTZ_MINDLIN") {
            return std::make_unique<EEPAContactModel>();
        }
        if (model_name == "JKR" || model_name == "jkr") {
            return std::make_unique<JKRContactModel>();
        }
        if (model_name == "None" || model_name == "none" || model_name.empty()) {
            return std::make_unique<NoContactModel>();
        }
        return nullptr;
    }

    static std::unique_ptr<ContactModel> create_eepa(float kn, float kt, float gamma_n, float gamma_t) {
        return std::make_unique<EEPAContactModel>(kn, kt, gamma_n, gamma_t);
    }

    static std::unique_ptr<ContactModel> create_jkr(float work_of_adhesion, float contact_radius0) {
        return std::make_unique<JKRContactModel>(work_of_adhesion, contact_radius0);
    }
};

// Device-side lightweight contact model views for CUDA kernels
struct DeviceEEPAContactView {
    float kn, kt, gamma_n, gamma_t;

    HOST_DEVICE_QUALIFIER float normal_force(float overlap, float velocity) const {
        return kn * overlap + gamma_n * velocity;
    }

    HOST_DEVICE_QUALIFIER float tangential_force(float tangential_overlap, float tangential_velocity) const {
        return kt * tangential_overlap + gamma_t * tangential_velocity;
    }
};

struct DeviceJKRContactView {
    float work_of_adhesion, contact_radius0;

    HOST_DEVICE_QUALIFIER float normal_force(float overlap, float velocity) const {
        float radius = contact_radius0 + overlap * 0.1f;
        return 6.0f * 3.14159f * work_of_adhesion * radius;
    }

    HOST_DEVICE_QUALIFIER float tangential_force(float tangential_overlap, float tangential_velocity) const {
        return 0.3f * normal_force(0, 0) * tangential_overlap;
    }
};

} // namespace matgpu