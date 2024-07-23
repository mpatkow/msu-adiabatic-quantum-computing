from qiskit_ibm_runtime import QiskitRuntimeService
 
# Save an IBM Quantum account and set it as your default account.
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="380f9153366a03bdf6083a73b7ddbd064b652615c6a8f192c0b64ea8a8aca2f7f7c500cafbb89f8e695a2cac4420da98e39e976c2743e2a843b605bc4577764a",
    set_as_default=True,
    # Use `overwrite=True` if you're updating your token.
    overwrite=True,
)
 
# Load saved credentials
service = QiskitRuntimeService()
