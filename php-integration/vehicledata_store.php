<?php
/**
 * VehicleData API — Imagin8 Internal Vehicle Data Store
 * 
 * Accepts vehicle identifiers (mmcode, vin, reg, clientRef) and stores them
 * in the VehicleData table, linking mmcode to vin for internal lookups.
 * Checks for existing records by vin or mmcode before inserting.
 *
 * Usage:
 *   curl -X POST https://www.evalue8.co.za/evalue8webservice/vehicledata_store.php \
 *     -H "Content-Type: application/json" \
 *     -d '{"mmcode":"00815004","vin":"WUAZZZFX2H7904038","reg":"HBH682K","clientRef":12345}'
 */

header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Headers: Content-Type");
header("Access-Control-Allow-Methods: POST, OPTIONS");

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

// Allow both GET and POST
if (!in_array($_SERVER['REQUEST_METHOD'], ['GET', 'POST'])) {
    http_response_code(405);
    echo json_encode(["successful" => false, "errorMessage" => "Method not allowed. Use GET or POST."]);
    exit;
}

header('Content-Type: application/json; charset=utf-8');

// Support JSON body, form-data, and GET parameters
$input = json_decode(file_get_contents('php://input'), true);
if (empty($input)) {
    $input = !empty($_POST) ? $_POST : $_GET;
}

// === Database Connection ===
$DB_HOST1 = 'dedi1266.jnb1.host-h.net';
$DB_USER_ACCOUNT1 = 'imagin8c_Clnt';
$DB_USER_PASSWORD1 = 'XuREWna6Tk8';
$DB_INSTANCE1 = 'imagin8c_ImgUsers';

try {
    $DB1 = new PDO("mysql:host={$DB_HOST1};dbname={$DB_INSTANCE1}", $DB_USER_ACCOUNT1, $DB_USER_PASSWORD1);
    $DB1->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    error_log("VehicleData DB connection failed: " . $e->getMessage());
    http_response_code(500);
    echo json_encode([
        "successful" => false,
        "errorMessage" => "Database connection failed."
    ]);
    exit;
}

// === Input Fields ===
$mmcode    = trim($input['mmcode']    ?? '');
$vin       = trim($input['vin']       ?? '');
$reg       = trim($input['reg']       ?? '');
$clientRef = $input['clientRef']       ?? 0;

// === Validate: clientRef is required ===
if (empty($clientRef)) {
    http_response_code(400);
    echo json_encode([
        "successful"   => false,
        "errorMessage" => "'clientRef' is required."
    ]);
    exit;
}

// === Validate: both vin AND mmcode must be provided ===
if (empty($vin) || empty($mmcode)) {
    http_response_code(400);
    echo json_encode([
        "successful"   => false,
        "errorMessage" => "Both 'vin' and 'mmcode' are required to create a vehicle record."
    ]);
    exit;
}

// === Create table if it doesn't exist ===
$createTable = "CREATE TABLE IF NOT EXISTS VehicleData (
    id          INT(11) AUTO_INCREMENT PRIMARY KEY,
    mmcode      VARCHAR(20) DEFAULT NULL,
    vin         VARCHAR(20) DEFAULT NULL,
    reg         VARCHAR(20) DEFAULT NULL,
    clientRef   INT(10) DEFAULT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_mmcode_vin (mmcode, vin),
    INDEX idx_vin (vin),
    INDEX idx_mmcode (mmcode),
    INDEX idx_reg (reg)
)";

try {
    $DB1->exec($createTable);
} catch (PDOException $e) {
    // Table likely already exists, continue
}

// Ensure unique index exists (safe to run on existing table)
try {
    $DB1->exec("ALTER TABLE VehicleData ADD UNIQUE KEY uq_mmcode_vin (mmcode, vin)");
} catch (PDOException $e) {
    // Index already exists, continue
}

// === Check if this exact mmcode + vin combination already exists ===
$stmt = $DB1->prepare("SELECT * FROM VehicleData WHERE mmcode = :mmcode AND vin = :vin LIMIT 1");
$stmt->execute([':mmcode' => $mmcode, ':vin' => $vin]);
$existing = $stmt->fetch(PDO::FETCH_ASSOC);

if ($existing) {
    // Update reg/clientRef if they were empty before
    $updates = [];
    $params = [':id' => $existing['id']];

    if (empty($existing['reg']) && !empty($reg)) {
        $updates[] = "reg = :reg";
        $params[':reg'] = $reg;
    }
    if (empty($existing['clientRef']) && !empty($clientRef)) {
        $updates[] = "clientRef = :clientRef";
        $params[':clientRef'] = $clientRef;
    }

    if (!empty($updates)) {
        $sql = "UPDATE VehicleData SET " . implode(", ", $updates) . " WHERE id = :id";
        $stmt = $DB1->prepare($sql);
        $stmt->execute($params);

        $stmt = $DB1->prepare("SELECT * FROM VehicleData WHERE id = :id");
        $stmt->execute([':id' => $existing['id']]);
        $existing = $stmt->fetch(PDO::FETCH_ASSOC);
    }

    echo json_encode([
        "successful" => true,
        "action"     => "existing",
        "message"    => "This mmcode + vin combination already exists.",
        "data"       => $existing,
    ], JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    exit;
}

// === Insert new record ===
try {
    $stmt = $DB1->prepare("INSERT INTO VehicleData (mmcode, vin, reg, clientRef) 
                           VALUES (:mmcode, :vin, :reg, :clientRef)");
    $stmt->execute([
        ':mmcode'    => $mmcode ?: null,
        ':vin'       => $vin ?: null,
        ':reg'       => $reg ?: null,
        ':clientRef' => $clientRef ?: null,
    ]);

    $newId = $DB1->lastInsertId();

    echo json_encode([
        "successful" => true,
        "action"     => "inserted",
        "message"    => "New vehicle record created.",
        "data"       => [
            "id"        => $newId,
            "mmcode"    => $mmcode ?: null,
            "vin"       => $vin ?: null,
            "reg"       => $reg ?: null,
            "clientRef" => $clientRef ?: null,
        ],
    ], JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);

} catch (PDOException $e) {
    error_log("VehicleData insert failed: " . $e->getMessage());
    http_response_code(500);
    echo json_encode([
        "successful"   => false,
        "errorMessage" => "Failed to insert vehicle data: " . $e->getMessage()
    ]);
}
?>
