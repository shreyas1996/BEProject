<?php
if(isset($_POST['submit']))
{
    if(@getimagesize($_FILES['image']['tmp_name']) == FALSE)
	{
       # echo "<span class='image_select'>please select an image</span>";

    }
    else
	{
        $image = addslashes($_FILES['image']['tmp_name']);
        $name  = addslashes($_FILES['image']['name']);
        $image = file_get_contents($image);
        $image = base64_encode($image);
        #saveimage($name,$image);
        $uploaddir = 'uploads/'; //this is your local directory
        $uploadfile = $uploaddir . basename($_FILES['image']['name']);

        echo "<p>";

            if (move_uploaded_file($_FILES['image']['tmp_name'], $uploadfile)) 
			{// file uploaded and moved
			} 
           # else { //uploaded but not moved}

        echo "</p>";


    }
	$tmp = exec("C:/Users/krishnamoothy/AppData/Local/conda/conda/envs/tensorflow/python pred.py $uploadfile");
	 //$tmp = exec("C:/Python27/python s.py .$uploadfile");
	 echo '<h1>'.$tmp.'</h1>';
	//header('Location: s.py');
}

?>