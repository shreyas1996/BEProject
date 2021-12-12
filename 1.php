<?php
if(isset($_POST['train']))
{
	header('Location: t.php');
}
else if(isset($_POST['pred']))
{
	header('Location: img.html');
}
?>