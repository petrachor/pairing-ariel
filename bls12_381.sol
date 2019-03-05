pragma solidity ^0.4.14;

contract AirDropper {
    mapping(address => uint) public balances;
    address owner;

    function AirDropper() public {
        owner = msg.sender;
    }

    function fund(address oldAddress) public payable {
        balances[oldAddress] += msg.value;
    }

    function redeem(address oldAddress, uint8 v, bytes32 r, bytes32 s) public returns (bool) {
        if (ecrecover(keccak256(msg.sender), v, r, s) != oldAddress) { return false; } // Old address did not sign new one
        if (balances[oldAddress] == 0) { return false; }

        uint balance = balances[oldAddress];
        balances[oldAddress] = 0; 
        msg.sender.transfer(balance);

        return true;
    }
}

contract Group {
    function getZero() public returns (bytes);
    function getOne() public returns (bytes);

    function hashTo(bytes seed, bytes data) public returns (bytes);
}

contract AdditiveGroup is Group {
    function add(bytes a, bytes b) public returns (bytes);
    function negate(bytes a) public returns (bytes);
    function scalarMul(bytes a, bytes scalar) public returns (bytes);
}

contract MultiplicativeGroup is Group {
    function mul(bytes a, bytes b) public returns (bytes);
    function inverse(bytes a) public returns (bytes);
}

contract PairingGroup is MultiplicativeGroup {
    function pairing(bytes g1, bytes g2) public returns (bytes);
    function multiPairing(bytes[] g1s, bytes[] g2s) public returns (bytes);
}

contract BLS {
    
    function privateToPublic(bytes privateKey) public returns (bytes) {
        return G1.scalarMul(G1.getOne(), privateKey);
    }

    function sign(bytes message, bytes privateKey) public returns (bytes) {
        return G1.scalarMul(G1.hashTo(seed, message), privateKey);
    }

    function verify(bytes message, bytes signature, bytes publicKey) public returns (bool) {
         return pairing(G1.hashTo(seed, message), publicKey)
         == pairing(signature, G2.one())
    }
}
