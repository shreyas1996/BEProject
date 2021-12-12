<html>
	<body>
		<center>
			<form action="img.php" method="POST">
				<input type="file" name="image" accept=img/*>
				<br>
				<input type="submit">
			</form>
		</center>
	</body>
</html>

<?php
$target_dir = "uploads/";
$target_file = $target_dir . basename($_FILES["image"]["name"]);
echo $target_dir;
echo $target_file;
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));
// Check if image file is a actual image or fake image
if(isset($_POST["submit"])) {
    $check = getimagesize($_FILES["image"]["tmp_name"]);
    if($check !== false) {
        echo "File is an image - " . $check["mime"] . ".";
        $uploadOk = 1;
    } else {
        echo "File is not an image.";
        $uploadOk = 0;
    }
}
?>
<!--
	$python='python cnn.py'
	echo $python
-->