<?php
/*
$target_dir = "uploads/";
$target_file = basename($_FILES['image']['name']);
#echo $target_dir;
echo $target_file;
$uploadOk = 1;
/*$s = pathinfo($target_file);
echo $s;
#$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));
#echo $imageFileType;
// Check if image file is a actual image or fake image
/*if(isset($_POST['submit'])) {
    $check = getimagesize($_FILES['image']['tmp_name']);
    if($check !== false) {
        echo "File is an image - " . $check['mime'] . ".";
        $uploadOk = 1;
    } else {
        echo "File is not an image.";
        $uploadOk = 0;
    }
}
/*foreach ($_FILES["image"]["error"] as $key => $error) {
    if ($error == UPLOAD_ERR_OK) {
        $tmp_name = $_FILES["iamge"]["tmp_name"][$key];
        // basename() may prevent filesystem traversal attacks;
        // further validation/sanitation of the filename may be appropriate
        //$name = basename($_FILES["image"]["name"][$key]);
		if(isset($_POST['submit']))
		{
			 move_uploaded_file($target_file, "$target_dir");
		}

*/
