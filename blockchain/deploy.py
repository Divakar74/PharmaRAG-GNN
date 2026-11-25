import os
from web3 import Web3
from eth_account import Account
from solcx import compile_source
from dotenv import load_dotenv

load_dotenv()


def deploy_contract(web3_provider: str, private_key: str, contract_source_path: str, constructor_args: tuple = ()):
    """
    Deploy smart contract to blockchain.
    
    Args:
        web3_provider: RPC URL for the blockchain network
        private_key: Private key for the deployer wallet
        contract_source_path: Path to the Solidity contract file
        constructor_args: Optional constructor arguments
    
    Returns:
        dict: Contains 'contract_address' and 'transaction_hash'
    """
    if not private_key:
        raise ValueError("PRIVATE_KEY environment variable not set")
    
    w3 = Web3(Web3.HTTPProvider(web3_provider))
    if not w3.is_connected():
        raise ConnectionError(f"Failed to connect to blockchain at {web3_provider}")
    
    account = Account.from_key(private_key)
    
    # Compile the contract
    with open(contract_source_path, 'r') as f:
        source = f.read()
    
    compiled = compile_source(source, solc_version='0.8.20')['<stdin>:DDIAuditTrail']
    
    # Get chain ID
    chain_id = w3.eth.chain_id
    
    # Get nonce
    nonce = w3.eth.get_transaction_count(account.address)
    
    # Get gas price
    gas_price = w3.eth.gas_price
    
    # Build contract
    contract = w3.eth.contract(abi=compiled['abi'], bytecode=compiled['bin'])
    
    # Estimate gas
    try:
        gas_estimate = contract.constructor().estimate_gas()
    except:
        gas_estimate = 500000  # Default if estimation fails
    
    gas_limit = int(gas_estimate * 1.2)  # 20% buffer
    
    # Build transaction
    tx = contract.constructor(*constructor_args).build_transaction({
        'chainId': chain_id,
        'gas': gas_limit,
        'gasPrice': gas_price,
        'nonce': nonce,
        'from': account.address
    })
    
    # Sign transaction
    signed_tx = w3.eth.account.sign_transaction(tx, private_key)
    
    # Send transaction
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    
    # Wait for receipt
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    return {
        'contract_address': receipt.contractAddress,
        'transaction_hash': tx_hash.hex(),
        'block_number': receipt.blockNumber,
        'gas_used': receipt.gasUsed
    }


def main():
    """Deploy the DDI Audit Trail contract to Polygon Amoy testnet."""
    rpc_url = os.getenv('POLYGON_AMOY_RPC', 'https://rpc-amoy.polygon.technology')
    private_key = os.getenv('PRIVATE_KEY')
    
    if not private_key:
        print("Error: PRIVATE_KEY not set in environment")
        return
    
    contract_path = 'blockchain/DDIAuditTrail.sol'
    
    try:
        result = deploy_contract(rpc_url, private_key, contract_path)
        print(f"Contract deployed successfully!")
        print(f"Contract Address: {result['contract_address']}")
        print(f"Transaction Hash: {result['transaction_hash']}")
        print(f"Block Number: {result['block_number']}")
        print(f"Gas Used: {result['gas_used']}")
    except Exception as e:
        print(f"Deployment failed: {e}")


if __name__ == '__main__':
    main()
