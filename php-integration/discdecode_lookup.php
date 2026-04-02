<?php
/**
 * DocDecode API Integration (Imagin8)
 * 
 * Accepts an image + lookuptype, calls the DocDecode /extract-doc API,
 * logs the transaction, and returns JSON.
 *
 * Supported lookuptype values:
 *   disc, licence, id, registration, invoice, generic
 *
 * Usage:
 *   curl -X POST https://www.evalue8.co.za/evalue8webservice/discdecode_lookup.php \
 *     -F "image=@disc.jpeg" \
 *     -F "lookuptype=disc" \
 *     -F "clientRef=12345" \
 *     -F "uname=rigard" \
 *     -F "comid=PC001" \
 *     -F "entityID=1"
 */

header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Headers: Content-Type");
header("Access-Control-Allow-Methods: POST, OPTIONS");

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

header('Content-Type: application/json; charset=utf-8');

// === DEBUG MODE ===
$DEBUG = false;

// === DocDecode API CONFIGURATION ===
$DISCDECODE_API_URL = "https://apidiscextract-production.up.railway.app";
$DISCDECODE_API_KEY = "disc_live_5e6ecb8bc69cae8e25643b045f59b2fca3a5d3eec12321451ea0aa34ab56baa3";

// === Database Connection (matching existing TransUnion integration) ===
$DB_HOST1 = 'dedi1266.jnb1.host-h.net';
$DB_USER_ACCOUNT1 = 'imagin8c_Clnt';
$DB_USER_PASSWORD1 = 'XuREWna6Tk8';
$DB_INSTANCE1 = 'imagin8c_ImgUsers';

try {
    $DB1 = new PDO("mysql:host={$DB_HOST1};dbname={$DB_INSTANCE1}", $DB_USER_ACCOUNT1, $DB_USER_PASSWORD1);
    $DB1->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    error_log("DB connection failed: " . $e->getMessage());
    $DB1 = null;
}

// === INPUT FIELDS (multipart form-data) ===
$clientRef       = $_POST['clientRef']       ?? '';
$lookupType      = $_POST['lookuptype']      ?? 'disc';
$computerName    = $_POST['comid']           ?? '';
$appUsername      = $_POST['uname']           ?? '';
$password         = $_POST['password']        ?? '';
$entityID         = $_POST['entityID']        ?? 0;
$requestorPerson  = $_POST['requestorPerson'] ?? '';

// === Map lookuptype to DocDecode doc_type ===
$docTypeMap = [
    'disc'                  => 'licence_disc',
    'licence_disc'          => 'licence_disc',
    'licence'               => 'drivers_licence',
    'drivers_licence'       => 'drivers_licence',
    'id'                    => 'id_document',
    'id_document'           => 'id_document',
    'registration'          => 'vehicle_registration',
    'vehicle_registration'  => 'vehicle_registration',
    'invoice'               => 'invoice',
    'generic'               => 'generic',
];
$docType = $docTypeMap[$lookupType] ?? 'generic';

$date = date('Y-m-d H:i:s');
$software_app = "DocDecodeApp";
$remoteAddress = $_SERVER['REMOTE_ADDR'] ?? '';
$request_uri   = $_SERVER['REQUEST_URI'] ?? '';

// === Validate required input ===
if (empty($clientRef)) {
    http_response_code(400);
    echo json_encode([
        "successful"   => false,
        "errorMessage" => "'clientRef' is required."
    ]);
    exit;
}

if (!isset($_FILES['image']) || $_FILES['image']['error'] !== UPLOAD_ERR_OK) {
    http_response_code(400);
    $uploadError = isset($_FILES['image']) ? $_FILES['image']['error'] : 'no file';
    echo json_encode([
        "successful"   => false,
        "errorMessage" => "Image file is required. Send as 'image' in multipart form-data. (error: $uploadError)"
    ]);
    exit;
}

$imageFile = $_FILES['image'];
$allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
$fileMimeType = mime_content_type($imageFile['tmp_name']);

if (!in_array($fileMimeType, $allowedTypes)) {
    http_response_code(400);
    echo json_encode([
        "successful"   => false,
        "errorMessage" => "Invalid image type. Allowed: JPEG, PNG, WebP, GIF. Got: {$fileMimeType}"
    ]);
    exit;
}

// === Call DocDecode API via /extract-doc ===
$ch = curl_init();

$cFile = new CURLFile(
    $imageFile['tmp_name'],
    $fileMimeType,
    $imageFile['name']
);

curl_setopt_array($ch, [
    CURLOPT_URL            => $DISCDECODE_API_URL . "/extract-doc",
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_POST           => true,
    CURLOPT_HTTPHEADER     => [
        "X-API-Key: {$DISCDECODE_API_KEY}",
    ],
    CURLOPT_POSTFIELDS     => [
        'image'    => $cFile,
        'doc_type' => $docType,
    ],
    CURLOPT_TIMEOUT        => 120,
    CURLOPT_SSL_VERIFYPEER => true,
    CURLOPT_SSL_VERIFYHOST => 2,
]);

$response = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$error    = curl_error($ch);
curl_close($ch);

// === Handle cURL errors ===
if ($error) {
    http_response_code(500);
    $output = [
        "successful"   => false,
        "errorMessage" => "Connection error to DocDecode API: $error"
    ];
    if ($DEBUG) {
        $output["debug"] = [
            "endpoint"  => $DISCDECODE_API_URL . "/extract-doc",
            "docType"   => $docType,
            "httpCode"  => 0,
            "curlError" => $error,
        ];
    }
    echo json_encode($output, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    exit;
}

// === Parse API response ===
$apiResponse = json_decode($response, true);

if ($apiResponse === null) {
    http_response_code(502);
    $output = [
        "successful"   => false,
        "errorMessage" => "Invalid response from DocDecode API (HTTP $httpCode)."
    ];
    if ($DEBUG) {
        $output["debug"] = [
            "httpCode"    => $httpCode,
            "rawResponse" => $response,
        ];
    }
    echo json_encode($output, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    exit;
}

// === Handle non-200 responses ===
if ($httpCode !== 200) {
    http_response_code(200);
    $output = [
        "successful"   => false,
        "errorMessage" => $apiResponse['error'] ?? $apiResponse['detail'] ?? "DocDecode API returned HTTP $httpCode."
    ];
    if ($DEBUG) {
        $output["debug"] = [
            "httpCode"    => $httpCode,
            "apiResponse" => $apiResponse,
        ];
    }
    echo json_encode($output, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    exit;
}

// === Extract data from successful response ===
$extractedData = $apiResponse['data'] ?? [];
$transactionId = $apiResponse['transaction_id'] ?? '';
$usage         = $apiResponse['usage'] ?? [];

$lookup_result = !empty($extractedData) ? 1 : 0;
$not_billable  = $lookup_result ? 0 : 1;

// === Map extracted fields per document type ===
$vin             = '';
$regNumber       = '';
$searchKeys      = '';
$searchType      = '';

switch ($docType) {
    case 'licence_disc':
        $vin         = $extractedData['vin'] ?? '';
        $regNumber   = $extractedData['vehicle_register_number'] ?? '';
        $searchKeys  = $vin ?: $regNumber;
        $searchType  = 'LicenceDisc';
        break;
    case 'drivers_licence':
        $searchKeys  = $extractedData['id_number'] ?? $extractedData['licence_number'] ?? '';
        $searchType  = 'DriversLicence';
        break;
    case 'id_document':
        $searchKeys  = $extractedData['id_number'] ?? '';
        $searchType  = 'IDDocument';
        break;
    case 'vehicle_registration':
        $vin         = $extractedData['vin'] ?? '';
        $regNumber   = $extractedData['registration_number'] ?? '';
        $searchKeys  = $vin ?: $regNumber;
        $searchType  = 'VehicleReg';
        break;
    case 'invoice':
        $searchKeys  = $extractedData['invoice_number'] ?? '';
        $searchType  = 'Invoice';
        break;
    default:
        $searchKeys  = $extractedData['title'] ?? 'generic';
        $searchType  = 'Generic';
        break;
}

// === Transaction Log ===
$transLogStatus = "not_attempted";

try {
    if ($DB1 === null) {
        throw new Exception("DB1 is null — connection failed at startup");
    }

    $transLogStatus = "db_connected";

    $insertQuery = "INSERT INTO Transaction_log 
        (computerName, AppUsername, mmCode, clientRef, vehicle_year, remote_address, 
         software_application, request_uri, log_date, lookup_result, product_id, 
         not_billable, SearchKeys, SearchType, SPEntityID, TransactionStatus, IM8TransRef,
         lookup_type, vin, reg)
        VALUES (:compName, :appUser, 0, :clientRef, 0, :address, :softApp, :requestUri, 
         :log_date, :lookupResult, :prodId, :notBillable, :searchKeys, :searchType, :entity, 'P', '',
         :lookupType, :vin, :reg)";

    $stmt = $DB1->prepare($insertQuery);
    $stmt->execute([
        ':compName'      => $computerName,
        ':appUser'       => $appUsername,
        ':clientRef'     => $clientRef,
        ':address'       => $remoteAddress,
        ':softApp'       => $software_app,
        ':requestUri'    => $request_uri,
        ':log_date'      => $date,
        ':lookupResult'  => $lookup_result,
        ':prodId'        => "400",
        ':notBillable'   => $not_billable,
        ':searchKeys'    => $searchKeys,
        ':searchType'    => $searchType,
        ':entity'        => $entityID,
        ':lookupType'    => $searchType,
        ':vin'           => $vin,
        ':reg'           => $regNumber,
    ]);

    $logIdRef = $DB1->lastInsertId();

    $updateQuery = "UPDATE Transaction_log 
        SET TransactionStatus = :transStatus, IM8TransRef = :IM8TransRef 
        WHERE log_id = :logId";

    $upd = $DB1->prepare($updateQuery);
    $upd->execute([
        ':transStatus' => 'D',
        ':IM8TransRef' => "DD" . str_pad($logIdRef, 9, '0', STR_PAD_LEFT),
        ':logId'       => $logIdRef,
    ]);

    $transLogStatus = "success (log_id: {$logIdRef})";

} catch (Exception $ex) {
    $transLogStatus = "FAILED: " . $ex->getMessage();
    error_log("DocDecode transaction log insert failed: " . $ex->getMessage());
}

// === Return JSON Response ===
http_response_code(200);
$output = [
    "successful"     => true,
    "lookupType"     => $docType,
    "transactionId"  => $transactionId,
    "data"           => $extractedData,
];

if ($DEBUG) {
    $output["debug"] = [
        "endpoint"       => $DISCDECODE_API_URL . "/extract-doc",
        "docType"        => $docType,
        "httpCode"       => $httpCode,
        "transactionLog" => $transLogStatus,
    ];
}

echo json_encode($output, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
?>
